#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <list>
#include <map>
#include <numeric>
#include <onnxruntime_cxx_api.h>

class BasisBoundaryFunction : public dealii::Function<3, double> {
public:
  BasisBoundaryFunction(dealii::DoFHandler<3>::active_cell_iterator cell, const unsigned int bf_index)
      : cell_(cell), bf_index_(bf_index) {}

  virtual double value(const dealii::Point<3> &p, const unsigned int component = 0) const override {
    dealii::Quadrature<3> quadrature(
        dealii::GeometryInfo<3>::project_to_unit_cell(dealii::StaticMappingQ1<3>::mapping.transform_real_to_unit_cell(cell_, p)));
    dealii::FEValues<3> fe_values(cell_->get_fe(), quadrature, dealii::update_values);

    fe_values.reinit(cell_);

    return fe_values.shape_value(bf_index_, component);
  }

private:
  dealii::DoFHandler<3>::active_cell_iterator cell_;
  unsigned int bf_index_;
};

class DCProblem {
public:
  DCProblem();
  void declare_parameters(const char *);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system_ms();
  void solve();
  void output_results() const;

  dealii::ParameterHandler prm;

  std::vector<dealii::Point<3>> sources;
  std::vector<double> currents;
  std::vector<dealii::Point<3>> sites;

  dealii::Triangulation<3> coarse_tria, tria;
  dealii::FE_Q<3> fe;
  dealii::DoFHandler<3> coarse_dof_handler;

  dealii::AffineConstraints<double> constraints;
  dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

  dealii::PETScWrappers::MPI::Vector solution;
  dealii::PETScWrappers::MPI::Vector system_rhs;

  std::unique_ptr<Ort::Env> ort_env;
  std::unique_ptr<Ort::Session> ort_session;
  std::unique_ptr<Ort::SessionOptions> ort_session_options;
  Ort::MemoryInfo memory_info;
  
  std::vector<float> input_rho_buffer;
  std::vector<float> input_x_buffer;
  std::vector<float> input_y_buffer;
  std::vector<float> input_z_buffer;
  
  std::vector<int64_t> input_shape;
  std::vector<int64_t> coord_shape;
};

DCProblem::DCProblem() : fe(1), coarse_dof_handler(coarse_tria),
                         memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DCProblem");
  
  ort_session_options = std::make_unique<Ort::SessionOptions>();
  ort_session_options->SetIntraOpNumThreads(0);
  ort_session_options->SetInterOpNumThreads(0);
  
  ort_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  ort_session_options->EnableMemPattern();
  ort_session_options->EnableCpuMemArena();
  
  const char *model_path = "model/model.onnx";
  ort_session = std::make_unique<Ort::Session>(*ort_env, model_path, *ort_session_options);
  
  input_rho_buffer.reserve(1024);
  input_x_buffer.reserve(1024);
  input_y_buffer.reserve(1024);
  input_z_buffer.reserve(1024);
  
  input_shape.resize(2);
  coord_shape.resize(2);
  
  std::vector<float> warmup_rho(64, 2.0f);
  std::vector<float> warmup_coords(125, 0.5f);
  input_shape[0] = 1; input_shape[1] = 64;
  coord_shape[0] = 1; coord_shape[1] = 125;
  
  std::vector<Ort::Value> warmup_tensors;
  warmup_tensors.reserve(4);
  warmup_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info, warmup_rho.data(), warmup_rho.size(), input_shape.data(), 2));
  for (int i = 0; i < 3; ++i) {
    warmup_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, warmup_coords.data(), warmup_coords.size(), coord_shape.data(), 2));
  }
  
  static const char* warmup_input_names[] = {"x1", "X", "Y", "Z"};
  static const char* warmup_output_names[] = {"output"};
  auto warmup_output = ort_session->Run(Ort::RunOptions{nullptr}, 
      warmup_input_names, warmup_tensors.data(), 4, warmup_output_names, 1);
  
  std::cout << "ONNX model loaded successfully" << std::endl;
}

void DCProblem::declare_parameters(const char *fn) {
  prm.declare_entry("iprefix", "", dealii::Patterns::FileName(), "Prefix of the input files");
  prm.declare_entry("oprefix", "", dealii::Patterns::FileName(dealii::Patterns::FileName::output), "Prefix of the output files");
  prm.parse_input(fn);
}

void DCProblem::make_grid() {
  {
    std::ifstream ifs_tria(prm.get("iprefix") + ".tria");

    if (!ifs_tria.good()) {
      abort();
    }
    boost::archive::binary_iarchive ia(ifs_tria);

    tria.clear();
    tria.load(ia, 0);
  }

  {
    std::ifstream ifs_tria(prm.get("iprefix") + "-coarse.tria");

    if (!ifs_tria.good()) {
      abort();
    }
    boost::archive::binary_iarchive ia(ifs_tria);

    coarse_tria.clear();
    coarse_tria.load(ia, 0);
  }

  std::ifstream ifs_emd(prm.get("iprefix") + ".emd");
  int nsources;
  ifs_emd >> nsources;
  sources.clear();
  currents.clear();
  for (int i = 0; i < nsources; ++i) {
    double x, y, z, current;
    ifs_emd >> x >> y >> z >> current;
    sources.push_back(dealii::Point<3>(x, y, z));
    currents.push_back(current);
  }

  int nsites;
  ifs_emd >> nsites;
  sites.clear();
  for (int i = 0; i < nsites; ++i) {
    double x, y, z;
    ifs_emd >> x >> y >> z;
    sites.push_back(dealii::Point<3>(x, y, z));
  }

  std::cout << "Number of active cells: " << tria.n_active_cells() << std::endl;
}

void DCProblem::setup_system() {
  coarse_dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << coarse_dof_handler.n_dofs() << std::endl;

  constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints(coarse_dof_handler, constraints);
  constraints.close();

  dealii::DynamicSparsityPattern dsp(coarse_dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(coarse_dof_handler, dsp, constraints);
  dsp.compress();

  system_matrix.reinit(coarse_dof_handler.locally_owned_dofs(), coarse_dof_handler.locally_owned_dofs(), dsp, MPI_COMM_SELF);

  solution.reinit(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  system_rhs.reinit(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
}

void DCProblem::assemble_system_ms() {
  dealii::QGauss<3> quadrature_formula(fe.degree + 1);
  dealii::FEValues<3> fe_values(fe, quadrature_formula,
                                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

  dealii::QGauss<2> face_quadrature_formula(fe.degree + 1);
  dealii::FEFaceValues<3> fe_face_values(fe, face_quadrature_formula,
                                         dealii::update_values | dealii::update_quadrature_points |
                                             dealii::update_normal_vectors | dealii::update_JxW_values);

  dealii::FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);

  std::list<std::pair<dealii::Triangulation<3>::cell_iterator, dealii::Triangulation<3>::cell_iterator>> list_cell_pair =
      dealii::GridTools::get_finest_common_cells(coarse_tria, tria);

  for (std::pair<dealii::Triangulation<3>::cell_iterator, dealii::Triangulation<3>::cell_iterator> cell_pair : list_cell_pair) {
    dealii::Triangulation<3>::cell_iterator coarse_cell_in_fine_tria = cell_pair.second;
    dealii::Triangulation<3>::active_cell_iterator coarse_cell(cell_pair.first);

    dealii::DoFHandler<3>::active_cell_iterator coarse_cell_dh =
        dealii::DoFHandler<3>::active_cell_iterator(*coarse_cell, &coarse_dof_handler);

    double sigma_coarse = 1.0 / coarse_cell_dh->material_id();

    cell_matrix = 0.0;

    if (!coarse_cell_in_fine_tria->has_children()) {
      fe_values.reinit(coarse_cell_dh);

      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            cell_matrix(i, j) +=
                sigma_coarse * (fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
      }
    } else {
      std::vector<dealii::Triangulation<3>::active_cell_iterator> child_cells =
      dealii::GridTools::get_active_child_cells<dealii::Triangulation<3>>(coarse_cell_in_fine_tria);

      std::map<dealii::Triangulation<3>::active_cell_iterator, unsigned int> material_id_map;
      for (auto cell : child_cells) {
        unsigned int mat_id = cell->material_id();
        material_id_map[cell] = mat_id;
      }

      dealii::Triangulation<3> local_tria;
      std::map<dealii::Triangulation<3>::active_cell_iterator, dealii::Triangulation<3>::active_cell_iterator> patch_map;
      dealii::GridTools::build_triangulation_from_patch<dealii::Triangulation<3>>(child_cells, local_tria, patch_map);

      std::map<dealii::Triangulation<3>::active_cell_iterator, unsigned int> local_material_id_map;
      for (auto cell = local_tria.begin_active(); cell != local_tria.end(); ++cell) {
        for (auto const& mapped_pair : patch_map) {
          if (mapped_pair.first == cell) {
            auto orig_material_id_it = material_id_map.find(mapped_pair.second);
            if (orig_material_id_it != material_id_map.end()) {
              cell->set_material_id(orig_material_id_it->second);
              local_material_id_map[cell] = orig_material_id_it->second;
            } else {
              cell->set_material_id(coarse_cell_dh->material_id());
              local_material_id_map[cell] = coarse_cell_dh->material_id();
            }
            break;
          }
        }
      }

      dealii::DoFHandler<3> local_dof_handler(local_tria);
      local_dof_handler.distribute_dofs(fe);

      dealii::AffineConstraints<double> zero_constraints;
      zero_constraints.clear();
      dealii::DoFTools::make_hanging_node_constraints(local_dof_handler, zero_constraints);
      dealii::VectorTools::interpolate_boundary_values(local_dof_handler, 0,
                                                      dealii::Functions::ConstantFunction<3, double>(0.0, 1), zero_constraints);
      zero_constraints.close();

      dealii::DynamicSparsityPattern dsp(local_dof_handler.n_dofs());
      dealii::DoFTools::make_sparsity_pattern(local_dof_handler, dsp, zero_constraints);
      dsp.compress();

      std::vector<dealii::PETScWrappers::MPI::Vector> local_solution(fe.dofs_per_cell);
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
        local_solution[i].reinit(local_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
      }

      dealii::QGauss<3> local_quadrature_formula(fe.degree + 1);
      dealii::FEValues<3> local_fe_values(fe, local_quadrature_formula,
                                          dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
      std::vector<dealii::types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

      input_rho_buffer.clear();
      input_x_buffer.clear();
      input_y_buffer.clear();
      input_z_buffer.clear();
      
      size_t n_cells = local_tria.n_active_cells();
      input_rho_buffer.reserve(n_cells);
      for (const auto &cell : local_dof_handler.active_cell_iterators()) {
          input_rho_buffer.push_back(log10f(static_cast<float>(cell->material_id())));
      }
      
      std::map<dealii::types::global_dof_index, dealii::Point<3>> support_points;
      dealii::MappingQ<3> mapping(1);
      dealii::DoFTools::map_dofs_to_support_points(mapping, local_dof_handler, support_points);
      
      size_t n_points = support_points.size();
      input_x_buffer.reserve(n_points);
      input_y_buffer.reserve(n_points);
      input_z_buffer.reserve(n_points);
      
      for (const auto &point_pair : support_points) {
          dealii::Point<3> rf_point = mapping.transform_real_to_unit_cell(coarse_cell_dh, point_pair.second);
          input_x_buffer.push_back(static_cast<float>(rf_point[0]));
          input_y_buffer.push_back(static_cast<float>(rf_point[1]));
          input_z_buffer.push_back(static_cast<float>(rf_point[2]));
      }
      
      input_shape[0] = 1; input_shape[1] = static_cast<int64_t>(input_rho_buffer.size());
      coord_shape[0] = 1; coord_shape[1] = static_cast<int64_t>(n_points);
      
      std::vector<Ort::Value> input_tensors;
      input_tensors.reserve(4);
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input_rho_buffer.data(), input_rho_buffer.size(), 
          input_shape.data(), 2));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input_x_buffer.data(), input_x_buffer.size(), 
          coord_shape.data(), 2));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input_y_buffer.data(), input_y_buffer.size(), 
          coord_shape.data(), 2));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input_z_buffer.data(), input_z_buffer.size(), 
          coord_shape.data(), 2));

      static const char* input_names[] = {"x1", "X", "Y", "Z"};
      static const char* output_names[] = {"output"};

      auto start_time = std::chrono::high_resolution_clock::now();

      auto output_tensors = ort_session->Run(
          Ort::RunOptions{nullptr},
          input_names,
          input_tensors.data(),
          input_tensors.size(),
          output_names,
          1);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(end_time - start_time);
      std::cout << "Block execution time: " << std::fixed << std::setprecision(6)
          << duration.count() << " seconds" << std::endl;

      float* output_data = output_tensors[0].GetTensorMutableData<float>();
      size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

      const size_t values_per_cell = output_size / fe.dofs_per_cell;
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
          float* cell_output = output_data + i * values_per_cell;
          size_t copy_size = std::min(values_per_cell, static_cast<size_t>(local_solution[i].size()));
          
          for (unsigned int j = 0; j < copy_size; ++j) {
            float val = cell_output[j];
            local_solution[i][j] = (val < 0.0009f && val != 0.0f) ? 0.0 : static_cast<double>(val);
          }
          local_solution[i].compress(dealii::VectorOperation::insert);
      }

      for (dealii::DoFHandler<3>::active_cell_iterator cell : local_dof_handler.active_cell_iterators()) {
        unsigned int material_id = cell->material_id();
        double sigma = 1.0 / material_id;
        local_fe_values.reinit(cell);
        for (const unsigned int q_index : local_fe_values.quadrature_point_indices()) {
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
              dealii::Tensor<1,3> grad_phi_i;
              dealii::Tensor<1,3> grad_phi_j;
              std::vector<dealii::types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int k = 0; k < fe.dofs_per_cell; ++k) {
                double scalar_i = local_solution[i](local_dof_indices[k]);
                double scalar_j = local_solution[j](local_dof_indices[k]);
                grad_phi_i += scalar_i * local_fe_values.shape_grad(k, q_index);
                grad_phi_j += scalar_j * local_fe_values.shape_grad(k, q_index);
              }
              cell_matrix(i,j) += sigma * (grad_phi_i * grad_phi_j) * 
                                  local_fe_values.JxW(q_index);
            }
          }
        }
      }
    }

    for (unsigned int f : coarse_cell_dh->face_indices()) {
      auto face = coarse_cell_dh->face(f);
      if (!face->at_boundary()) {
        continue;
      }

      fe_face_values.reinit(coarse_cell_dh, f);
      if (fe_face_values.get_normal_vectors()[0][2] < 0) {
        continue;
      }

      for (size_t src_idx = 0; src_idx < sources.size(); ++src_idx) {
        dealii::Tensor<1, 3> r = face->center() - sources[src_idx];
        for (unsigned int q_index : fe_face_values.quadrature_point_indices()) {
          dealii::Tensor<1, 3> n = fe_face_values.normal_vector(q_index);
          double foo = sigma_coarse * dealii::scalar_product(r, n) / (r.norm() * n.norm() * r.norm());
          for (unsigned int i : fe_face_values.dof_indices()) {
            for (unsigned int j : fe_face_values.dof_indices()) {
              cell_matrix(i, j) += (foo * fe_face_values.shape_value(i, q_index) * fe_face_values.shape_value(j, q_index) *
                                    fe_face_values.JxW(q_index));
            }
          }
        }
      }
    }

    std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
    coarse_cell_dh->get_dof_indices(dof_indices);

    constraints.distribute_local_to_global(cell_matrix, dof_indices, system_matrix);
  }

  system_matrix.compress(dealii::VectorOperation::add);

  system_rhs = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    dealii::Vector<double> tmp_v(coarse_dof_handler.n_dofs());
    dealii::VectorTools::create_point_source_vector(coarse_dof_handler, sources[i], tmp_v);
    if (i < currents.size()) {
      tmp_v *= currents[i];
    }

    dealii::PETScWrappers::MPI::Vector tmp_petsc(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
    tmp_petsc = tmp_v;
    system_rhs += tmp_petsc;
  }
}

void DCProblem::solve() {
  dealii::SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());
  dealii::PETScWrappers::SolverCG solver(solver_control, MPI_COMM_SELF);
  dealii::PETScWrappers::PreconditionBoomerAMG preconditioner(system_matrix);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}

void DCProblem::output_results() const {
  dealii::Vector<double> rho(tria.n_active_cells());
  for (auto cell : tria.active_cell_iterators()) {
    rho[cell->active_cell_index()] = cell->material_id();
  }

  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(coarse_dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.build_patches();

  std::ofstream output(prm.get("oprefix") + "-solution.vtk");
  data_out.write_vtk(output);

  std::vector<double> u(sites.size());

  for (int i = 0; i < (int)sites.size(); ++i) {
    u[i] = dealii::VectorTools::point_value(coarse_dof_handler, solution, sites[i]);
  }

  std::ofstream ofs_rsp(prm.get("oprefix") + ".rsp");
  for (int i = 0; i < (int)sites.size(); ++i) {
    ofs_rsp << sites[i] << " " << u[i] << std::endl;
  }
}

void DCProblem::run() {
  auto start_time_all = std::chrono::high_resolution_clock::now();

  make_grid();
  setup_system();

  auto start_time_ass = std::chrono::high_resolution_clock::now();
  assemble_system_ms();
  auto end_time_ass = std::chrono::high_resolution_clock::now();
  auto duration_ass = std::chrono::duration<double>(end_time_ass - start_time_ass);
  std::cout << "Assemble system execution time: " << std::fixed << std::setprecision(6) << duration_ass.count() << " seconds" << std::endl;

  auto start_time_solve = std::chrono::high_resolution_clock::now();
  solve();
  auto end_time_solve = std::chrono::high_resolution_clock::now();
  auto duration_solve = std::chrono::duration<double>(end_time_solve - start_time_solve);
  std::cout << "Solve execution time: " << std::fixed << std::setprecision(6) << duration_solve.count() << " seconds" << std::endl;

  auto end_time_all = std::chrono::high_resolution_clock::now();
  auto duration_all = std::chrono::duration<double>(end_time_all - start_time_all);
  std::cout << "All execution time: " << std::fixed << std::setprecision(6) << duration_all.count() << " seconds" << std::endl;

  output_results();
}

int main(int argc, char **argv) {
  dealii::Utilities::MPI::MPI_InitFinalize init(argc, argv, 1);

  if (argc < 2) {
    std::cout << "Usage: <parameter file>" << std::endl;
    return 0;
  }

  DCProblem dc;
  dc.declare_parameters(argv[1]);
  dc.run();

  return 0;
}
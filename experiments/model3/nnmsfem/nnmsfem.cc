#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

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
  void simulate_wenner_line();
  void output_results() const;

  dealii::ParameterHandler prm;

  std::vector<dealii::Point<3>> sources;
  std::vector<double> currents;
  std::vector<dealii::Point<3>> sites;

  // std::vector<double> rho;

  dealii::Triangulation<3> coarse_tria, tria;
  dealii::FE_Q<3> fe;
  dealii::DoFHandler<3> coarse_dof_handler;

  dealii::AffineConstraints<double> constraints;
  dealii::PETScWrappers::MPI::SparseMatrix stiffness_matrix; // volume terms only, reused across Wenner configs
  dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

  dealii::PETScWrappers::MPI::Vector solution;
  dealii::PETScWrappers::MPI::Vector system_rhs;
};

DCProblem::DCProblem() : fe(1), coarse_dof_handler(coarse_tria) {}

void DCProblem::declare_parameters(const char *fn) {
  prm.declare_entry("iprefix", "", dealii::Patterns::FileName(), "Prefix of the input files");
  prm.declare_entry("oprefix", "", dealii::Patterns::FileName(dealii::Patterns::FileName::output), "Prefix of the output files");
  prm.declare_entry("wenner_spacing", "10.0", dealii::Patterns::Double(0), "Legacy single electrode spacing a for Wenner array");
  prm.declare_entry("wenner_spacings", "32,48,64,80,96,112,128,144,160", dealii::Patterns::List(dealii::Patterns::Double(0)),
                    "Comma-separated electrode spacings a for Wenner array (multiple depths)");
  prm.declare_entry("wenner_start_x", "-320.0", dealii::Patterns::Double(), "Starting x-position for Wenner array shift (A electrode)");
  prm.declare_entry("wenner_end_x", "320.0", dealii::Patterns::Double(), "Ending x-position for Wenner array shift (B electrode must stay before this)");
  prm.declare_entry("wenner_current", "10.0", dealii::Patterns::Double(0), "Injection current magnitude for Wenner array");
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

  // std::ifstream ifs_rho(prm.get("iprefix") + ".rho");
  // int nrhos;
  // ifs_rho >> nrhos;
  // for (int i = 0; i < nrhos; ++i) {
  //   double r;
  //   ifs_rho >> r;
  //   rho.push_back(r);
  // }

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
  stiffness_matrix.reinit(coarse_dof_handler.locally_owned_dofs(), coarse_dof_handler.locally_owned_dofs(), dsp, MPI_COMM_SELF);

  solution.reinit(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  system_rhs.reinit(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
}

void DCProblem::assemble_system_ms() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  const char *model_path = "model/model.onnx";
  Ort::Session session(env, model_path, session_options);
  // std::cout << "ONNX models loaded!" << std::endl;

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

    // double sigma_coarse = 1.0 / rho[coarse_cell_dh->material_id()];
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
      auto start_time = std::chrono::high_resolution_clock::now();

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

      std::vector<float> input_rho;
      for (const auto &cell : local_dof_handler.active_cell_iterators()) {
          // 对电阻率进行log10处理
          float resistivity = static_cast<float>(cell->material_id());
          input_rho.push_back(std::log10(resistivity));
      }
      std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_rho.size())};
      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info,
          input_rho.data(),
          input_rho.size(),
          input_shape.data(),
          input_shape.size()
      );

      std::vector<float> input_x, input_y, input_z;
      std::map<dealii::types::global_dof_index, dealii::Point<3>> support_points;
      dealii::MappingQ<3> mapping(1);
      dealii::DoFTools::map_dofs_to_support_points(mapping, local_dof_handler, support_points);
      for (const auto &point_pair : support_points) {
          const dealii::Point<3> &point = point_pair.second;
          dealii::Point<3> rf_point = mapping.transform_real_to_unit_cell(coarse_cell_dh, point);
          input_x.push_back(static_cast<float>(rf_point[0]));
          input_y.push_back(static_cast<float>(rf_point[1]));
          input_z.push_back(static_cast<float>(rf_point[2]));
      }
      std::vector<int64_t> coord_shape = {1, static_cast<int64_t>(input_x.size())};
      Ort::Value input_tensor_x = Ort::Value::CreateTensor<float>(
          memory_info, input_x.data(), input_x.size(), coord_shape.data(), coord_shape.size());
      Ort::Value input_tensor_y = Ort::Value::CreateTensor<float>(
          memory_info, input_y.data(), input_y.size(), coord_shape.data(), coord_shape.size());
      Ort::Value input_tensor_z = Ort::Value::CreateTensor<float>(
          memory_info, input_z.data(), input_z.size(), coord_shape.data(), coord_shape.size());
      std::vector<Ort::Value> input_tensors;
      input_tensors.push_back(std::move(input_tensor));
      input_tensors.push_back(std::move(input_tensor_x));
      input_tensors.push_back(std::move(input_tensor_y)); 
      input_tensors.push_back(std::move(input_tensor_z));

      const char* input_names[] = {"x1", "X", "Y", "Z"};
      const char* output_names[] = {"output"};

      auto output_tensors = session.Run(
          Ort::RunOptions{nullptr},
          input_names,
          input_tensors.data(),
          input_tensors.size(),
          output_names,
          1);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(end_time - start_time);
      std::cout << "Else block execution time: " << std::fixed << std::setprecision(6)
          << duration.count() << " seconds" << std::endl;

      float* output_data = output_tensors[0].GetTensorMutableData<float>();
      size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
      std::cout << "Output size: " << output_size << std::endl;

      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
          std::vector<float> solution_values(output_size/fe.dofs_per_cell);
          std::copy(output_data + i*125, output_data + (i+1)*125, solution_values.begin());
          for (unsigned int j = 0; j < solution_values.size() && j < local_solution[i].size(); ++j) {
            if (solution_values[j] < 0.0009 && solution_values[j] != 0) {
              local_solution[i][j] = -0.0;
            } else {
              local_solution[i][j] = solution_values[j];
            }
          }
          local_solution[i].compress(dealii::VectorOperation::insert);
      }

      // for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
      //     std::vector<float> solution_values(output_size/fe.dofs_per_cell);
      //     std::copy(output_data + i*125, output_data + (i+1)*125, solution_values.begin());
      //     for (unsigned int j = 0; j < solution_values.size() && j < local_solution[i].size(); ++j) {
      //         local_solution[i][j] = solution_values[j];
      //     }
      //     local_solution[i].compress(dealii::VectorOperation::insert);
      // }

      // dealii::DataOut<3> data_out;
      // data_out.attach_dof_handler(local_dof_handler);
      // for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
      //   data_out.add_data_vector(local_solution[i], "u" + dealii::Utilities::int_to_string(i));
      // }
      // data_out.build_patches();
      // std::ofstream output(prm.get("oprefix") + "-" + coarse_cell->id().to_string() + ".vtk");
      // data_out.write_vtk(output);

      // std::ofstream solution_file("solution.txt", std::ios::app);
      // if (!solution_file.is_open()) {
      //   std::cerr << "Failed to open solution.txt for writing" << std::endl;
      // }
      // for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
      //   if (solution_file.is_open()) {
      //     for (unsigned int j = 0; j < local_solution[i].size(); ++j) {
      //       solution_file << local_solution[i][j] << (j < local_solution[i].size() - 1 ? "," : "");
      //     }
      //     solution_file << std::endl;
      //   }
      // }
      // if (solution_file.is_open()) {
      //   solution_file.close();
      //   std::cout << "Saved all local solutions to solution.txt" << std::endl;
      // }

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

    // std::ofstream cell_matrix_file("cell_matrix_dcd_nn.txt", std::ios::app);
    // if (cell_matrix_file.is_open()) {
    //   for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
    //     for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
    //       cell_matrix_file << cell_matrix(i, j);
    //       if (j < fe.dofs_per_cell - 1) {
    //         cell_matrix_file << ",";
    //       }
    //     }
    //     cell_matrix_file << std::endl;
    //   }
    //   cell_matrix_file.close();
    //   std::cout << "Saved local system matrix to cell_matrix_dcd.txt" << std::endl;
    // } else {
    //   std::cerr << "Failed to open cell_matrix_dcd.txt for writing" << std::endl;
    // }

    for (unsigned int f : coarse_cell_dh->face_indices()) {
      auto face = coarse_cell_dh->face(f);
      if (!face->at_boundary()) {
        continue;
      }

      fe_face_values.reinit(coarse_cell_dh, f);
      if (fe_face_values.get_normal_vectors()[0][2] < 0) {
        continue;
      }

      // NOTE: boundary Robin terms depend on the current Wenner electrode
      // positions (A and B), which change for every configuration.  They are
      // therefore NOT assembled here; instead they are added per-configuration
      // inside simulate_wenner_line() after copying stiffness_matrix into
      // system_matrix.
    }

    std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
    coarse_cell_dh->get_dof_indices(dof_indices);

    constraints.distribute_local_to_global(cell_matrix, dof_indices, stiffness_matrix);
  }

  stiffness_matrix.compress(dealii::VectorOperation::add);
  // The right-hand side and full system_matrix (stiffness + Robin BC) are
  // built later for each Wenner configuration.
  system_rhs = 0;
}

void DCProblem::solve() {
  dealii::SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());
  dealii::PETScWrappers::SolverCG solver(solver_control, MPI_COMM_SELF);
  dealii::PETScWrappers::PreconditionBoomerAMG preconditioner(system_matrix);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}

void DCProblem::simulate_wenner_line() {
  std::vector<double> spacings;
  {
    const std::string spacing_str = prm.get("wenner_spacings");
    const auto tokens = dealii::Utilities::split_string_list(spacing_str);
    for (const auto &tok : tokens) {
      if (!tok.empty()) {
        spacings.push_back(std::stod(tok));
      }
    }
    if (spacings.empty()) {
      spacings.push_back(prm.get_double("wenner_spacing"));
    }
  }

  const double start_x = prm.get_double("wenner_start_x");
  const double end_x   = prm.get_double("wenner_end_x");
  const double current = prm.get_double("wenner_current");

  if (current <= 0.0) {
    throw std::runtime_error("wenner_current must be positive");
  }

  // Prepare face quadrature (reused across all configurations).
  dealii::QGauss<2> face_quadrature_formula(fe.degree + 1);
  dealii::FEFaceValues<3> fe_face_values(fe, face_quadrature_formula,
                                         dealii::update_values | dealii::update_quadrature_points |
                                             dealii::update_normal_vectors | dealii::update_JxW_values);
  dealii::FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

  std::ofstream ofs(prm.get("oprefix") + "-wenner-rho.txt");
  ofs << "# a center_x A_x B_x M_x N_x rho_app\n";

  unsigned int configuration_count = 0;

  for (const double a : spacings) {
    if (a <= 0.0) {
      throw std::runtime_error("All wenner_spacings must be positive");
    }

    for (double Ax = start_x; Ax + 3.0 * a <= end_x + 1e-10; Ax += a) {
      // const dealii::Point<3> A(Ax, 0.0, 0.0);
      // const dealii::Point<3> M(Ax + a, 0.0, 0.0);
      // const dealii::Point<3> N(Ax + 2.0 * a, 0.0, 0.0);
      // const dealii::Point<3> B(Ax + 3.0 * a, 0.0, 0.0);

      const dealii::Point<3> A(0.0, Ax, 0.0);
      const dealii::Point<3> M(0.0, Ax + a, 0.0);
      const dealii::Point<3> N(0.0, Ax + 2.0 * a, 0.0);
      const dealii::Point<3> B(0.0, Ax + 3.0 * a, 0.0);

      // Current sources for this Wenner configuration: +I at A, -I at B.
      const std::vector<dealii::Point<3>> cfg_sources = {A, B};

      // Rebuild system_matrix = stiffness_matrix + boundary Robin terms for A and B.
      system_matrix.copy_from(stiffness_matrix);

      for (const auto &coarse_cell_dh : coarse_dof_handler.active_cell_iterators()) {
        double sigma_coarse = 1.0 / coarse_cell_dh->material_id();
        cell_matrix = 0.0;

        for (unsigned int f : coarse_cell_dh->face_indices()) {
          auto face = coarse_cell_dh->face(f);
          if (!face->at_boundary()) continue;

          fe_face_values.reinit(coarse_cell_dh, f);
          if (fe_face_values.get_normal_vectors()[0][2] < 0) continue;

          for (const auto &src : cfg_sources) {
            dealii::Tensor<1, 3> r = face->center() - src;
            for (unsigned int q_index : fe_face_values.quadrature_point_indices()) {
              dealii::Tensor<1, 3> n = fe_face_values.normal_vector(q_index);
              double foo = sigma_coarse * dealii::scalar_product(r, n) / (r.norm() * n.norm() * r.norm());
              for (unsigned int i : fe_face_values.dof_indices())
                for (unsigned int j : fe_face_values.dof_indices())
                  cell_matrix(i, j) += foo * fe_face_values.shape_value(i, q_index) *
                                       fe_face_values.shape_value(j, q_index) *
                                       fe_face_values.JxW(q_index);
            }
          }
        }

        coarse_cell_dh->get_dof_indices(dof_indices);
        constraints.distribute_local_to_global(cell_matrix, dof_indices, system_matrix);
      }
      system_matrix.compress(dealii::VectorOperation::add);

      // Build RHS: point sources at A (+I) and B (-I).
      system_rhs = 0;

      dealii::Vector<double> tmp_A(coarse_dof_handler.n_dofs());
      dealii::VectorTools::create_point_source_vector(coarse_dof_handler, A, tmp_A);
      tmp_A *= current;
      dealii::PETScWrappers::MPI::Vector tmp_A_p(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
      tmp_A_p = tmp_A;
      system_rhs += tmp_A_p;

      dealii::Vector<double> tmp_B(coarse_dof_handler.n_dofs());
      dealii::VectorTools::create_point_source_vector(coarse_dof_handler, B, tmp_B);
      tmp_B *= -current;
      dealii::PETScWrappers::MPI::Vector tmp_B_p(coarse_dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
      tmp_B_p = tmp_B;
      system_rhs += tmp_B_p;

      solve();

      const double uM = dealii::VectorTools::point_value(coarse_dof_handler, solution, M);
      const double uN = dealii::VectorTools::point_value(coarse_dof_handler, solution, N);
      const double deltaV  = uM - uN;
      const double rho_app = 2.0 * dealii::numbers::PI * a * deltaV / current;
      const double center_x = Ax + 1.5 * a;

      ofs << a << " " << center_x << " " << Ax << " " << Ax + 3.0 * a
          << " " << Ax + a << " " << Ax + 2.0 * a << " " << rho_app << std::endl;

      ++configuration_count;
    }
  }

  std::cout << "Computed " << configuration_count << " Wenner apparent resistivities." << std::endl;
}

void DCProblem::output_results() const {
  dealii::Vector<double> rho(tria.n_active_cells());
  for (auto cell : tria.active_cell_iterators()) {
    // rho[cell->active_cell_index()] = this->rho[cell->material_id()];
    rho[cell->active_cell_index()] = cell->material_id();
  }

  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(coarse_dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.build_patches();

  std::ofstream output(prm.get("oprefix") + "-solution.vtk");
  data_out.write_vtk(output);

  if (!sites.empty()) {
    std::vector<double> u(sites.size());
    for (int i = 0; i < (int)sites.size(); ++i) {
      u[i] = dealii::VectorTools::point_value(coarse_dof_handler, solution, sites[i]);
    }
    std::ofstream ofs_rsp(prm.get("oprefix") + ".rsp");
    for (int i = 0; i < (int)sites.size(); ++i) {
      ofs_rsp << sites[i] << " " << u[i] << std::endl;
    }
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
  simulate_wenner_line();
  auto end_time_solve = std::chrono::high_resolution_clock::now();
  auto duration_solve = std::chrono::duration<double>(end_time_solve - start_time_solve);
  std::cout << "Wenner sweep execution time: " << std::fixed << std::setprecision(6) << duration_solve.count() << " seconds" << std::endl;

  auto end_time_all = std::chrono::high_resolution_clock::now();
  auto duration_all = std::chrono::duration<double>(end_time_all - start_time_all);
  std::cout << "All execution time: " << std::fixed << std::setprecision(6) << duration_all.count() << " seconds" << std::endl;

  output_results();
}

int main(int argc, char **argv) {
  dealii::Utilities::MPI::MPI_InitFinalize init(argc, argv, 1);

  if (argc < 2) {
    std::cout << "Usage: dcfwd3d <parameter file>" << std::endl;
    return 0;
  }

  DCProblem dc;
  dc.declare_parameters(argv[1]);
  dc.run();

  return 0;
}
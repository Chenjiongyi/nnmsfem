#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
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
#include <fstream>
#include <numeric>

class DCProblem {
public:
  DCProblem();
  void declare_parameters(const char *);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  dealii::ParameterHandler prm;

  std::vector<dealii::Point<3>> sources;
  std::vector<double> currents;
  std::vector<dealii::Point<3>> sites;

  dealii::Triangulation<3> triangulation;
  dealii::FE_Q<3> fe;
  dealii::DoFHandler<3> dof_handler;

  dealii::AffineConstraints<double> constraints;
  dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

  dealii::PETScWrappers::MPI::Vector solution;
  dealii::PETScWrappers::MPI::Vector system_rhs;
};

DCProblem::DCProblem() : fe(1), dof_handler(triangulation) {}

void DCProblem::declare_parameters(const char *fn) {
  prm.declare_entry("iprefix", "", dealii::Patterns::FileName(), "Prefix of the input files");
  prm.declare_entry("oprefix", "", dealii::Patterns::FileName(dealii::Patterns::FileName::output), "Prefix of the output files");
  prm.parse_input(fn);
}

void DCProblem::make_grid() {
  std::ifstream ifs_tria(prm.get("iprefix") + ".tria");

  if (!ifs_tria.good()) {
    abort();
  }
  boost::archive::binary_iarchive ia(ifs_tria);

  triangulation.clear();
  triangulation.load(ia, 0);

  std::ifstream ifs_emd(prm.get("iprefix") + ".emd");
  
  int nsources;
  ifs_emd >> nsources;
  for (int i = 0; i < nsources; ++i) {
    double x, y, z, current;
    ifs_emd >> x >> y >> z >> current;
    sources.push_back(dealii::Point<3>(x, y, z));
    currents.push_back(current);
  }

  int nsites;
  ifs_emd >> nsites;
  for (int i = 0; i < nsites; ++i) {
    double x, y, z;
    ifs_emd >> x >> y >> z;
    sites.push_back(dealii::Point<3>(x, y, z));
  }

  std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

void DCProblem::setup_system() {
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  dsp.compress();

  system_matrix.reinit(dof_handler.locally_owned_dofs(), dof_handler.locally_owned_dofs(), dsp, MPI_COMM_SELF);

  solution.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  system_rhs.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
}

void DCProblem::assemble_system() {
  dealii::QGauss<3> quadrature_formula(fe.degree + 1);
  dealii::FEValues<3> fe_values(fe, quadrature_formula,
                                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

  dealii::QGauss<2> face_quadrature_formula(fe.degree + 1);
  dealii::FEFaceValues<3> fe_face_values(fe, face_quadrature_formula,
                                         dealii::update_values | dealii::update_quadrature_points |
                                             dealii::update_normal_vectors | dealii::update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    double sigma = 1.0 / cell->material_id();

    fe_values.reinit(cell);

    cell_matrix = 0;
    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) += sigma * (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                                        fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                                        fe_values.JxW(q_index));           // dx
    }
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int f : cell->face_indices()) {
      auto face = cell->face(f);
      if (!face->at_boundary()) {
        continue;
      }

      fe_face_values.reinit(cell, f);
      if (fe_face_values.get_normal_vectors()[0][2] < 0) {
        continue;
      }

      for (size_t src_idx = 0; src_idx < sources.size(); ++src_idx) {
        dealii::Tensor<1, 3> r = face->center() - sources[src_idx];
        for (unsigned int q_index : fe_face_values.quadrature_point_indices()) {
          dealii::Tensor<1, 3> n = fe_face_values.normal_vector(q_index);
          double foo = sigma * dealii::scalar_product(r, n) / (r.norm() * n.norm() * r.norm());
          for (unsigned int i : fe_face_values.dof_indices()) {
            for (unsigned int j : fe_face_values.dof_indices()) {
              cell_matrix(i, j) += (foo * fe_face_values.shape_value(i, q_index) * fe_face_values.shape_value(j, q_index) *
                                    fe_face_values.JxW(q_index));
            }
          }
        }
      }
    }

    constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
  }

  system_matrix.compress(dealii::VectorOperation::add);

  system_rhs = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    dealii::Vector<double> tmp_v(dof_handler.n_dofs());
    dealii::VectorTools::create_point_source_vector(dof_handler, sources[i], tmp_v);
    tmp_v *= currents[i];
    
    dealii::PETScWrappers::MPI::Vector tmp_petsc(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
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
  dealii::Vector<double> rho(triangulation.n_active_cells());
  for (auto cell : triangulation.active_cell_iterators()) {
    rho[cell->active_cell_index()] = cell->material_id();
  }

  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.add_data_vector(rho, "rho");
  data_out.build_patches();

  std::ofstream output(prm.get("oprefix") + "-solution.vtk");
  data_out.write_vtk(output);

  std::vector<double> u(sites.size());

  for (int i = 0; i < (int)sites.size(); ++i) {
    u[i] = dealii::VectorTools::point_value(dof_handler, solution, sites[i]);
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
  assemble_system();
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
  dealii::Utilities::MPI::MPI_InitFinalize init(argc, argv);

  if (argc < 2) {
    std::cout << "Usage: <parameter file>" << std::endl;
    return 0;
  }

  DCProblem dc;
  dc.declare_parameters(argv[1]);
  dc.run();

  return 0;
}
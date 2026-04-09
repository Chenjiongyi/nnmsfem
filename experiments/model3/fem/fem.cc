#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

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

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/archive_exception.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>

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
  void simulate_wenner_line();
  void output_results() const;

  dealii::ParameterHandler prm;

  std::vector<dealii::Point<3>> sources;
  std::vector<double> currents;
  std::vector<dealii::Point<3>> sites;

  // std::vector<double> rho;

  dealii::Triangulation<3> triangulation;
  dealii::FE_Q<3> fe;
  dealii::DoFHandler<3> dof_handler;

  dealii::AffineConstraints<double> constraints;
  dealii::PETScWrappers::MPI::SparseMatrix stiffness_matrix; // volume terms only
  dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

  dealii::PETScWrappers::MPI::Vector solution;
  dealii::PETScWrappers::MPI::Vector system_rhs;
};

DCProblem::DCProblem() : fe(1), dof_handler(triangulation) {}

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

// Helper: load triangulation from binary archive, falling back to no_header mode
// if boost version mismatch causes an unsupported_version exception.
static void load_tria(dealii::Triangulation<3> &tria, const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.good()) {
    std::cerr << "Cannot open triangulation file: " << path << std::endl;
    abort();
  }

  // First attempt: normal load (archive header present and versions match).
  try {
    boost::archive::binary_iarchive ia(ifs);
    tria.clear();
    tria.load(ia, 0);
    return;
  } catch (const boost::archive::archive_exception &e) {
    std::cerr << "[load_tria] Normal load failed (" << e.what()
              << "), retrying with no_header flag." << std::endl;
  }

  // Second attempt: skip the boost archive header (version check bypassed).
  // Works when the file was written with a different boost version but the
  // underlying data layout is still compatible.
  ifs.clear();
  ifs.seekg(0, std::ios::beg);
  {
    boost::archive::binary_iarchive ia(ifs, boost::archive::no_header);
    tria.clear();
    tria.load(ia, 0);
  }
}

void DCProblem::make_grid() {
  load_tria(triangulation, prm.get("iprefix") + ".tria");

  // std::ifstream ifs_rho(prm.get("iprefix") + ".rho");
  // int nrhos;
  // ifs_rho >> nrhos;
  // for (int i = 0; i < nrhos; ++i) {
  //   double r;
  //   ifs_rho >> r;
  //   rho.push_back(r);
  // }

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
  stiffness_matrix.reinit(dof_handler.locally_owned_dofs(), dof_handler.locally_owned_dofs(), dsp, MPI_COMM_SELF);

  solution.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
  system_rhs.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
}

void DCProblem::assemble_system() {
  dealii::QGauss<3> quadrature_formula(fe.degree + 1);
  dealii::FEValues<3> fe_values(fe, quadrature_formula,
                                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Assemble volume (stiffness) terms only.
  // Boundary Robin terms depend on the current source positions and are
  // added per Wenner configuration inside simulate_wenner_line().
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    double sigma = 1.0 / cell->material_id();

    fe_values.reinit(cell);
    cell_matrix = 0;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) += sigma * (fe_values.shape_grad(i, q_index) *
                                        fe_values.shape_grad(j, q_index) *
                                        fe_values.JxW(q_index));

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix, local_dof_indices, stiffness_matrix);
  }

  stiffness_matrix.compress(dealii::VectorOperation::add);
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
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

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
      const std::vector<double>           cfg_currents = {current, -current};

      // Build system_matrix = stiffness_matrix + boundary Robin terms for A and B.
      system_matrix.copy_from(stiffness_matrix);

      for (const auto &cell : dof_handler.active_cell_iterators()) {
        double sigma = 1.0 / cell->material_id();
        cell_matrix = 0;

        for (unsigned int f : cell->face_indices()) {
          auto face = cell->face(f);
          if (!face->at_boundary()) continue;

          fe_face_values.reinit(cell, f);
          if (fe_face_values.get_normal_vectors()[0][2] < 0) continue;

          for (size_t src_idx = 0; src_idx < cfg_sources.size(); ++src_idx) {
            dealii::Tensor<1, 3> r = face->center() - cfg_sources[src_idx];
            for (unsigned int q_index : fe_face_values.quadrature_point_indices()) {
              dealii::Tensor<1, 3> n = fe_face_values.normal_vector(q_index);
              double foo = sigma * dealii::scalar_product(r, n) / (r.norm() * n.norm() * r.norm());
              for (unsigned int i : fe_face_values.dof_indices())
                for (unsigned int j : fe_face_values.dof_indices())
                  cell_matrix(i, j) += foo * fe_face_values.shape_value(i, q_index) *
                                       fe_face_values.shape_value(j, q_index) *
                                       fe_face_values.JxW(q_index);
            }
          }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
      }
      system_matrix.compress(dealii::VectorOperation::add);

      // Build RHS: point sources at A (+I) and B (-I).
      system_rhs = 0;
      for (size_t i = 0; i < cfg_sources.size(); ++i) {
        dealii::Vector<double> tmp(dof_handler.n_dofs());
        dealii::VectorTools::create_point_source_vector(dof_handler, cfg_sources[i], tmp);
        tmp *= cfg_currents[i];
        dealii::PETScWrappers::MPI::Vector tmp_p(dof_handler.locally_owned_dofs(), MPI_COMM_SELF);
        tmp_p = tmp;
        system_rhs += tmp_p;
      }

      solve();

      const double uM = dealii::VectorTools::point_value(dof_handler, solution, M);
      const double uN = dealii::VectorTools::point_value(dof_handler, solution, N);
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
  dealii::Vector<double> rho(triangulation.n_active_cells());
  for (auto cell : triangulation.active_cell_iterators()) {
    // rho[cell->active_cell_index()] = this->rho[cell->material_id()];
    rho[cell->active_cell_index()] = cell->material_id();
  }

  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.add_data_vector(rho, "rho");
  data_out.build_patches();

  std::ofstream output(prm.get("oprefix") + "-solution.vtk");
  data_out.write_vtk(output);

  if (!sites.empty()) {
    std::vector<double> u(sites.size());
    for (int i = 0; i < (int)sites.size(); ++i) {
      u[i] = dealii::VectorTools::point_value(dof_handler, solution, sites[i]);
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
  assemble_system();
  auto end_time_ass = std::chrono::high_resolution_clock::now();
  auto duration_ass = std::chrono::duration<double>(end_time_ass - start_time_ass);
  std::cout << "Assemble system execution time: " << std::fixed << std::setprecision(6) << duration_ass.count() << " seconds" << std::endl;

  auto start_time_solve = std::chrono::high_resolution_clock::now();
  simulate_wenner_line();
  auto end_time_solve = std::chrono::high_resolution_clock::now();
  auto duration_solve = std::chrono::duration<double>(end_time_solve - start_time_solve);
  std::cout << "Wenner sweep execution time: " << std::fixed << std::setprecision(6) << duration_solve.count() << " seconds" << std::endl;

  output_results();

  auto end_time_all = std::chrono::high_resolution_clock::now();
  auto duration_all = std::chrono::duration<double>(end_time_all - start_time_all);
  std::cout << "All execution time: " << std::fixed << std::setprecision(6) << duration_all.count() << " seconds" << std::endl;
}

int main(int argc, char **argv) {
  dealii::Utilities::MPI::MPI_InitFinalize init(argc, argv);

  if (argc < 2) {
    std::cout << "Usage: dcfwd3d <parameter file>" << std::endl;
    return 0;
  }

  DCProblem dc;
  dc.declare_parameters(argv[1]);
  dc.run();

  return 0;
}
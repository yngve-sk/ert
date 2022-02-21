#pragma once

#include <stdexcept>
#define HAVE_THREAD_POOL 1
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/ensemble_config.hpp>

#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_obs.hpp>

namespace analysis {
/**
 * Container for all data required for performing an update step.
 * Data consists of 5 matrices and a list of pairs of rowscaling and matrix.
 * objects mask describing the observations which
 * are active. In addition a flag has_observations which is used to determine wheter
 * it is possible to do an update step.
*/
class update_data_type : public std::enable_shared_from_this<update_data_type> {
public:
    update_data_type() = default;
    update_data_type(
        matrix_type *S_in, matrix_type *E_in, matrix_type *D_in,
        matrix_type *R_in, std::optional<Eigen::MatrixXd> A_in,
        std::vector<std::pair<matrix_type *, std::shared_ptr<RowScaling>>>
            A_with_rowscaling_in,
        const std::vector<bool> &obs_mask_in)
        : obs_mask(obs_mask_in) {
        S = *S_in;
        E = *E_in;
        D = *D_in;
        R = *R_in;
        A = A_in;
        A_with_rowscaling = A_with_rowscaling_in;
        has_observations = true;
    }

    Eigen::MatrixXd S;
    Eigen::MatrixXd E;
    Eigen::MatrixXd D;
    Eigen::MatrixXd R;
    std::optional<Eigen::MatrixXd> A;
    std::vector<bool> obs_mask;
    std::vector<std::pair<Eigen::MatrixXd *, std::shared_ptr<RowScaling>>>
        A_with_rowscaling;
    bool has_observations = false;
};

bool smoother_update(const local_updatestep_type *updatestep,
                     int total_ens_size, enkf_obs_type *obs,
                     rng_type *shared_rng,
                     const analysis_config_type *analysis_config,
                     ensemble_config_type *ensemble_config,
                     enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                     bool verbose);
} // namespace analysis

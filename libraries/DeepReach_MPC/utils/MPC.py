import torch
from tqdm import tqdm
import math


class MPC:
    def __init__(self, dT, horizon, receding_horizon, num_samples, dynamics_, device, mode="MPC",
                 sample_mode="gaussian", lambda_=0.01, style="direct", num_iterative_refinement=1):
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.receding_horizon = receding_horizon
        self.dynamics_ = dynamics_

        self.dT = dT

        self.lambda_ = lambda_

        self.mode = mode
        self.sample_mode = sample_mode
        self.style = style  # choice: receding, direct
        self.num_iterative_refinement = num_iterative_refinement
        self.num_effective_horizon_refinement = 0

    def get_batch_data(self, initial_condition_tensor, T, policy=None, t=0.0):
        '''
        Generate MPC dataset in a batch manner
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                T: MPC total horizon
                t: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                costs: best cost function for all initial_condition_tensor (A)
                state_trajs: best trajs for all initial_condition_tensor (A * Horizon * state_dim)
                coords: bootstrapped MPC coords after normalization (coords=[time,state])  (? * (state_dim+1))
                value_labels: bootstrapped MPC value labels  (?)
        '''
        self.T = T*1.0
        self.batch_size = initial_condition_tensor.shape[0]
        if self.dynamics_.set_mode in ['avoid', 'reach']:
            state_trajs, lxs, num_iters = self.get_opt_trajs(
                initial_condition_tensor, policy, t)
            costs, _ = torch.min(lxs, dim=-1)

        elif self.dynamics_.set_mode == 'reach_avoid':
            state_trajs, avoid_values, reach_values, num_iters = self.get_opt_trajs(
                initial_condition_tensor, policy, t)
            costs = torch.min(torch.maximum(
                reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values
        else:
            raise NotImplementedError

        # generating MPC dataset: {..., (t, x, J, u), ...} NEW
        coords = torch.empty(0, self.dynamics_.state_dim+1).to(self.device)
        value_labels = torch.empty(0).to(self.device)
        # bootstrapping will be accurate up until the min l(x) occur
        if self.dynamics_.set_mode in ['avoid', 'reach']:
            _, min_idx = torch.min(lxs, dim=-1)
        elif self.dynamics_.set_mode == 'reach_avoid':
            _, min_idx = torch.min(torch.clamp(
                reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)), dim=-1)
        for i in range(num_iters):
            coord_i = torch.zeros(
                self.batch_size, self.dynamics_.state_dim+1).to(self.device)
            coord_i[:, 0] = self.T - i * self.dT
            coord_i[:, 1:] = state_trajs[:, i, :]*1.0
            if self.dynamics_.set_mode in ['avoid', 'reach']:
                valid_idx = (min_idx > i).nonzero(as_tuple=True)
                value_labels_i = torch.min(
                    lxs[valid_idx[0], i:], dim=-1).values
                coord_i = coord_i[valid_idx]
            elif self.dynamics_.set_mode == 'reach_avoid':
                valid_idx = (min_idx > i).nonzero(as_tuple=True)
                value_labels_i = torch.min(torch.clamp(reach_values[valid_idx[0], i:], min=torch.max(
                    -avoid_values[valid_idx[0], i:], dim=-1).values.unsqueeze(-1)), dim=-1).values
                coord_i = coord_i[valid_idx]
            else:
                raise NotImplementedError
            # add to data
            coords = torch.cat((coords, coord_i), dim=0)
            value_labels = torch.cat((value_labels, value_labels_i), dim=0)

        ##################### only use in range labels ###################################################
        output1 = torch.all(coords[..., 1:] >= self.dynamics_.state_range_[
                            :, 0]-0.01, -1, keepdim=False)
        output2 = torch.all(coords[..., 1:] <= self.dynamics_.state_range_[
                            :, 1]+0.01, -1, keepdim=False)
        in_range_index = torch.logical_and(torch.logical_and(
            output1, output2), ~torch.isnan(value_labels))

        coords = coords[in_range_index]
        value_labels = value_labels[in_range_index]
        ###################################################################################################
        coords = self.dynamics_.coord_to_input(coords)

        return costs, state_trajs, coords.detach().cpu().clone(), value_labels.detach().cpu().clone()

    def get_opt_trajs(self, initial_condition_tensor, policy=None, t=0.0):
        '''
        Generate optimal trajs in a batch manner
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                t: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 

                best_trajs: best trajs for all initial_condition_tensor (A * Horizon * state_dim)
                lxs: l(x) along best trajs (A*H)
                num_iters: H 
        '''
        num_iters = math.ceil((self.T)/self.dT)
        self.horizon = math.ceil((self.T)/self.dT)

        self.incremental_horizon = math.ceil((self.T-t)/self.dT)
        if self.style == 'direct':

            self.init_control_tensors()
            if policy is not None:
                self.num_effective_horizon_refinement = int(
                    self.num_iterative_refinement*0.4)
                for i in range(self.num_effective_horizon_refinement):
                    # optimize on the effective horizon first
                    self.warm_start_with_policy(
                        initial_condition_tensor, policy, t)
            # optimize on the entire horizon for stability (in case that the current learned value function is not accurate)
            best_controls, best_trajs = self.get_control(
                initial_condition_tensor, self.num_iterative_refinement, policy, t_remaining=t)

            if self.dynamics_.set_mode in ['avoid', 'reach']:
                lxs = self.dynamics_.boundary_fn(best_trajs)
                return best_trajs, lxs, num_iters
            elif self.dynamics_.set_mode == 'reach_avoid':
                avoid_values = self.dynamics_.avoid_fn(best_trajs)
                reach_values = self.dynamics_.reach_fn(best_trajs)
                return best_trajs, avoid_values, reach_values, num_iters
            else:
                raise NotImplementedError

        elif self.style == 'receding':
            if self.dynamics_.set_mode == 'reach_avoid':
                raise NotImplementedError

            state_trajs = torch.zeros(
                (self.batch_size, num_iters+1, self.dynamics_.state_dim)).to(self.device)  # A*H*D
            state_trajs[:, 0, :] = initial_condition_tensor

            self.init_control_tensors()
            if policy is not None:
                self.num_effective_horizon_refinement = int(
                    self.num_iterative_refinement*0.4)
                for i in range(self.num_effective_horizon_refinement):
                    self.warm_start_with_policy(
                        initial_condition_tensor, policy, t)
            lxs = torch.zeros(self.batch_size, num_iters+1).to(self.device)

            for i in tqdm(range(int(num_iters/self.receding_horizon))):
                best_controls, _ = self.get_control(
                    state_trajs[:, i, :])
                for k in range(self.receding_horizon):
                    lxs[:, i*self.receding_horizon+k] = self.dynamics_.boundary_fn(
                        state_trajs[:, i*self.receding_horizon+k, :])
                    state_trajs[:, i*self.receding_horizon+1+k, :] = self.get_next_step_state(
                        state_trajs[:, i*self.receding_horizon+k, :], best_controls[:, k, :])
                    self.receiding_start += 1
            lxs[:, -1] = self.dynamics_.boundary_fn(state_trajs[:, -1, :])
            return state_trajs, lxs, num_iters
        else:
            return NotImplementedError

    def warm_start_with_policy(self, initial_condition_tensor, policy=None, t_remaining=None):
        '''
        Generate optimal trajs in a batch manner using the DeepReach value function as the terminal cost
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                t_remaining: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                None
                Internally update self.control_tensors (first H_R horizon with MPC and t_remaining with DeepReach policy)
                Internally update self.warm_start_traj (for debugging purpose)
        '''
        if self.incremental_horizon > 0:
            # Rollout with the incremental horizon
            state_trajs_H, permuted_controls_H, _ = self.rollout_dynamics(
                initial_condition_tensor, start_iter=0, rollout_horizon=self.incremental_horizon)

            costs = self.dynamics_.cost_fn(state_trajs_H)  # A * N
            # Use the learned value function for terminal cost and compute the cost function
            if t_remaining > 0.0:
                traj_times = torch.ones(self.batch_size, self.num_samples, 1).to(
                    self.device)*t_remaining
                state_trajs_clamped = torch.clamp(state_trajs_H[:, :, -1, :], torch.tensor(self.dynamics_.state_test_range(
                )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])

                traj_coords = torch.cat(
                    (traj_times, state_trajs_clamped), dim=-1)
                traj_policy_results = policy(
                    {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
                terminal_values = self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
                ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
                if self.incremental_horizon > 0:
                    costs = torch.minimum(costs, terminal_values)
                    if self.dynamics_.set_mode == 'reach_avoid':
                        avoid_value_max = torch.max(
                            -self.dynamics_.avoid_fn(state_trajs_H), dim=-1).values
                        costs = torch.maximum(costs, avoid_value_max)
                        # TODO: Fix the cost function computation for receding horizon MPC
                else:
                    costs = terminal_values*1.0
            # Pick the best control and correponding traj
            if self.dynamics_.set_mode == 'avoid':
                best_costs, best_idx = costs.max(1)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                best_costs, best_idx = costs.min(1)
            else:
                raise NotImplementedError
            expanded_idx = best_idx[..., None, None, None].expand(
                -1, -1, permuted_controls_H.size(2), permuted_controls_H.size(3))

            best_controls_H = torch.gather(
                permuted_controls_H, dim=1, index=expanded_idx).squeeze(1)  # A * H * D_u
            expanded_idx_traj = best_idx[..., None, None, None].expand(
                -1, -1, state_trajs_H.size(2), state_trajs_H.size(3))
            best_traj_H = torch.gather(
                state_trajs_H, dim=1, index=expanded_idx_traj).squeeze(1)

            # Rollout the remaining horizon with the learned policy and update the nominal control traj
            self.control_tensors[:, :self.incremental_horizon,
                                 :] = best_controls_H*1.0
            self.warm_start_traj = self.rollout_with_policy(
                best_traj_H[:, -1, :], policy, self.horizon-self.incremental_horizon, self.incremental_horizon)
            self.warm_start_traj = torch.cat(
                [best_traj_H[:, :-1, :], self.warm_start_traj], dim=1)

        else:
            # Rollout using the learned policy and update the nominal control traj
            self.warm_start_traj = self.rollout_with_policy(
                initial_condition_tensor, policy, self.horizon)

    def get_control(self, initial_condition_tensor, num_iterative_refinement=1, policy=None, t_remaining=None):
        '''
        Update self.control_tensors using perturbations
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                num_iterative_refinement: number of iterative improvement steps (re-sampling steps) in MPC
                t_remaining: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        '''

        if self.style == 'direct':
            if num_iterative_refinement == -1:  # rollout using the policy
                best_traj = self.rollout_with_policy(
                    initial_condition_tensor, policy, self.horizon)
            for i in range(num_iterative_refinement+1 - self.num_effective_horizon_refinement):
                state_trajs, permuted_controls, permuted_disturbances = self.rollout_dynamics(
                    initial_condition_tensor, start_iter=0, rollout_horizon=self.horizon)
                self.all_state_trajs = state_trajs.detach().cpu()*1.0
                _, best_traj, best_costs = self.update_control_tensor(
                    state_trajs, permuted_controls, permuted_disturbances)
            return self.control_tensors, best_traj
        elif self.style == 'receding':
            # initial_condition_tensor: A*D
            state_trajs, permuted_controls, permuted_disturbances = self.rollout_dynamics(
                initial_condition_tensor, start_iter=self.receiding_start, rollout_horizon=self.horizon-self.receiding_start)

            current_controls, best_traj, _ = self.update_control_tensor(
                state_trajs, permuted_controls)

            return current_controls, best_traj

    def rollout_with_policy(self, initial_condition_tensor, policy, policy_horizon, policy_start_iter=0):
        '''
        Rollout traj with policy and update nominal control/disturbance tensors.

        Inputs:
            initial_condition_tensor: A*D_N
            policy: Current DeepReach model
            policy_horizon: number of rollout steps
            policy_start_iter: offset into control/disturbance tensors
        '''
        state_trajs = torch.zeros(
            (self.batch_size, policy_horizon + 1, self.dynamics_.state_dim)
        )
        state_trajs = state_trajs.to(self.device, non_blocking=True)

        state_trajs[:, 0, :] = initial_condition_tensor * 1.0
        state_trajs_clamped = state_trajs * 1.0

        traj_times = torch.ones(self.batch_size, 1).to(
            self.device
        ) * policy_horizon * self.dT

        has_disturbance = self.dynamics_.disturbance_dim > 0

        for k in range(policy_horizon):
            tensor_idx = k + policy_start_iter

            traj_coords = torch.cat(
                (traj_times, state_trajs_clamped[:, k, :]),
                dim=-1,
            )

            traj_policy_results = policy(
                {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))}
            )

            traj_dvs = self.dynamics_.io_to_dv(
                traj_policy_results['model_in'],
                traj_policy_results['model_out'].squeeze(dim=-1),
            ).detach()

            state_k = traj_coords[:, 1:].to(self.device)
            dvds_k = traj_dvs[..., 1:].to(self.device)

            self.control_tensors[:, tensor_idx, :] = self.dynamics_.optimal_control(
                state_k,
                dvds_k,
            )

            self.control_tensors[:, tensor_idx, :] = self.dynamics_.clamp_control(
                state_trajs[:, k, :],
                self.control_tensors[:, tensor_idx, :],
            )

            if has_disturbance:
                self.disturbance_tensors[:, tensor_idx, :] = self.dynamics_.optimal_disturbance(
                    state_k,
                    dvds_k,
                )

                self.disturbance_tensors[:, tensor_idx, :] = self.dynamics_.clamp_disturbance(
                    state_trajs[:, k, :],
                    self.disturbance_tensors[:, tensor_idx, :],
                )

                disturbance_k = self.disturbance_tensors[:, tensor_idx, :]
            else:
                disturbance_k = None

            state_trajs[:, k + 1, :] = self.get_next_step_state(
                state_trajs[:, k, :],
                self.control_tensors[:, tensor_idx, :],
                disturbance_k,
            )

            state_trajs_clamped[:, k + 1, :] = torch.clamp(
                state_trajs[:, k + 1, :],
                torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 0],
                torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1],
            )

            traj_times = traj_times - self.dT

        return state_trajs


    def update_control_tensor(self, state_trajs, permuted_controls, permuted_disturbances=None):
        '''
        Determine nominal controls using sampled trajectories.

        Inputs:
            state_trajs: A*N*H*D_N
            permuted_controls: A*N*H*D_U
            permuted_disturbances: A*N*H*D_D or None

        For disturbance systems, N = num_control_samples * num_disturbance_samples.
        For avoid mode, this selects max_u min_d cost.
        '''
        costs = self.dynamics_.cost_fn(state_trajs)  # A * N

        if self.mode == "MPC":
            if permuted_disturbances is not None:
                num_control_samples = self.num_control_samples
                num_disturbance_samples = self.num_disturbance_samples

                costs_game = costs.reshape(
                    self.batch_size,
                    num_control_samples,
                    num_disturbance_samples,
                )

                if self.dynamics_.set_mode == 'avoid':
                    # Disturbance chooses the least safe outcome; control chooses
                    # the safest among those worst-case outcomes.
                    worst_costs, worst_dist_idx = costs_game.min(dim=2)
                    best_costs, best_control_idx = worst_costs.max(dim=1)
                elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                    # For reach-style objectives, lower cost is better for control,
                    # so the disturbance chooses the largest cost.
                    worst_costs, worst_dist_idx = costs_game.max(dim=2)
                    best_costs, best_control_idx = worst_costs.min(dim=1)
                else:
                    raise NotImplementedError

                batch_idx = torch.arange(self.batch_size, device=self.device)
                best_dist_idx = worst_dist_idx[batch_idx, best_control_idx]
                best_idx = best_control_idx * num_disturbance_samples + best_dist_idx

            else:
                # Original control-only behavior.
                if self.dynamics_.set_mode == 'avoid':
                    best_costs, best_idx = costs.max(1)
                elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                    best_costs, best_idx = costs.min(1)
                else:
                    raise NotImplementedError

            expanded_idx = best_idx[..., None, None, None].expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3)
            )

            best_controls = torch.gather(
                permuted_controls,
                dim=1,
                index=expanded_idx,
            ).squeeze(1)  # A * H * D_u

            if self.style == 'direct':
                self.control_tensors = best_controls * 1.0
            elif self.style == 'receding':
                self.control_tensors[:, self.receiding_start:, :] = best_controls * 1.0
            else:
                raise NotImplementedError

            if permuted_disturbances is not None:
                expanded_idx_dist = best_idx[..., None, None, None].expand(
                    -1, -1, permuted_disturbances.size(2), permuted_disturbances.size(3)
                )

                best_disturbances = torch.gather(
                    permuted_disturbances,
                    dim=1,
                    index=expanded_idx_dist,
                ).squeeze(1)  # A * H * D_d

                if self.style == 'direct':
                    self.disturbance_tensors = best_disturbances * 1.0
                elif self.style == 'receding':
                    self.disturbance_tensors[:, self.receiding_start:, :] = best_disturbances * 1.0
                else:
                    raise NotImplementedError

            expanded_idx_traj = best_idx[..., None, None, None].expand(
                -1, -1, state_trajs.size(2), state_trajs.size(3)
            )

            best_traj = torch.gather(
                state_trajs,
                dim=1,
                index=expanded_idx_traj,
            ).squeeze(1)

        elif self.mode == "MPPI":
            if permuted_disturbances is not None:
                raise NotImplementedError(
                    "Two-player disturbance MPPI is not implemented; use mode='MPC'."
                )

            # Original weighted-average control-only behavior.
            if self.dynamics_.set_mode == 'avoid':
                exp_terms = torch.exp((1 / self.lambda_) * costs)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                exp_terms = torch.exp((1 / self.lambda_) * -costs)
            else:
                raise NotImplementedError

            den = torch.sum(exp_terms, dim=-1)

            num = torch.sum(
                exp_terms[:, :, None, None].repeat(
                    1, 1, self.horizon, self.dynamics_.control_dim
                ) * permuted_controls,
                dim=1,
            )

            self.control_tensors = num / den[:, None, None]

            self.control_tensors = torch.clamp(
                self.control_tensors,
                self.dynamics_.control_range_[..., 0],
                self.dynamics_.control_range_[..., 1],
            )

            best_costs, best_idx = costs.max(1) if self.dynamics_.set_mode == 'avoid' else costs.min(1)
            expanded_idx_traj = best_idx[..., None, None, None].expand(
                -1, -1, state_trajs.size(2), state_trajs.size(3)
            )
            best_traj = torch.gather(
                state_trajs,
                dim=1,
                index=expanded_idx_traj,
            ).squeeze(1)

        else:
            raise NotImplementedError

        current_controls = self.control_tensors[
            :,
            self.receiding_start:self.receiding_start + self.receding_horizon,
            :
        ]

        return current_controls, best_traj, best_costs


    def rollout_nominal_trajs(self, initial_state_tensor):
        '''
        Rollout trajs with nominal controls (self.control_tensors)
        '''
        # rollout trajs
        state_trajs = torch.zeros(
            (self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs[:, 0, :] = initial_state_tensor*1.0  # A * D

        for k in range(self.horizon):

            state_trajs[:, k+1, :] = self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k, :])
        return state_trajs

    def rollout_dynamics(self, initial_state_tensor, start_iter, rollout_horizon, eps_var_factor=1):
        '''
        Rollout trajs by generating perturbed controls and, if present, disturbances.

        Outputs:
            state_trajs: A*N*H*D_N
            permuted_controls: A*N*H*D_U
            permuted_disturbances: A*N*H*D_D or None
        '''
        has_disturbance = self.dynamics_.disturbance_dim > 0

        if has_disturbance:
            num_control_samples = int(math.sqrt(self.num_samples))
            num_disturbance_samples = int(math.sqrt(self.num_samples))
            total_samples = num_control_samples * num_disturbance_samples
        else:
            num_control_samples = self.num_samples
            num_disturbance_samples = 1
            total_samples = self.num_samples

        self.num_control_samples = num_control_samples
        self.num_disturbance_samples = num_disturbance_samples

        if self.sample_mode == "gaussian":
            epsilon_tensor = torch.randn(
                self.batch_size,
                num_control_samples,
                rollout_horizon,
                self.dynamics_.control_dim,
                device=self.device,
            ) * torch.sqrt(self.dynamics_.eps_var).to(self.device) * eps_var_factor

            epsilon_tensor[:, 0, ...] = 0.0

            sampled_controls = (
                self.control_tensors[:, start_iter:start_iter + rollout_horizon, :]
                .unsqueeze(1)
                .repeat(1, num_control_samples, 1, 1)
                + epsilon_tensor
            )

        elif self.sample_mode == "binary":
            sampled_controls = torch.sign(torch.empty(
                self.batch_size,
                num_control_samples,
                rollout_horizon,
                self.dynamics_.control_dim,
                device=self.device,
            ).uniform_(-1, 1))

            sampled_controls[:, 0, ...] = (
                self.control_tensors[:, start_iter:start_iter + rollout_horizon, :] * 1.0
            )

        else:
            raise NotImplementedError

        sampled_controls = torch.clamp(
            sampled_controls,
            self.dynamics_.control_range_[..., 0],
            self.dynamics_.control_range_[..., 1],
        )

        if has_disturbance:
            disturbance_range = self.dynamics_.disturbance_range_
            disturbance_eps_var = getattr(
                self.dynamics_,
                "disturbance_eps_var",
                self.dynamics_.eps_var,
            )

            if self.sample_mode == "gaussian":
                disturbance_epsilon = torch.randn(
                    self.batch_size,
                    num_disturbance_samples,
                    rollout_horizon,
                    self.dynamics_.disturbance_dim,
                    device=self.device,
                ) * torch.sqrt(disturbance_eps_var).to(self.device) * eps_var_factor

                disturbance_epsilon[:, 0, ...] = 0.0

                sampled_disturbances = (
                    self.disturbance_tensors[:, start_iter:start_iter + rollout_horizon, :]
                    .unsqueeze(1)
                    .repeat(1, num_disturbance_samples, 1, 1)
                    + disturbance_epsilon
                )

            elif self.sample_mode == "binary":
                sampled_disturbances = torch.sign(torch.empty(
                    self.batch_size,
                    num_disturbance_samples,
                    rollout_horizon,
                    self.dynamics_.disturbance_dim,
                    device=self.device,
                ).uniform_(-1, 1))

                sampled_disturbances[:, 0, ...] = (
                    self.disturbance_tensors[:, start_iter:start_iter + rollout_horizon, :] * 1.0
                )

            sampled_disturbances = torch.clamp(
                sampled_disturbances,
                disturbance_range[..., 0],
                disturbance_range[..., 1],
            )

            permuted_controls = (
                sampled_controls[:, :, None, :, :]
                .repeat(1, 1, num_disturbance_samples, 1, 1)
                .reshape(
                    self.batch_size,
                    total_samples,
                    rollout_horizon,
                    self.dynamics_.control_dim,
                )
            )

            permuted_disturbances = (
                sampled_disturbances[:, None, :, :, :]
                .repeat(1, num_control_samples, 1, 1, 1)
                .reshape(
                    self.batch_size,
                    total_samples,
                    rollout_horizon,
                    self.dynamics_.disturbance_dim,
                )
            )

        else:
            permuted_controls = sampled_controls
            permuted_disturbances = None

        state_trajs = torch.zeros(
            (
                self.batch_size,
                total_samples,
                rollout_horizon + 1,
                self.dynamics_.state_dim,
            ),
            device=self.device,
        )

        state_trajs[:, :, 0, :] = initial_state_tensor.unsqueeze(
            1
        ).repeat(1, total_samples, 1)

        for k in range(rollout_horizon):
            permuted_controls[:, :, k, :] = self.dynamics_.clamp_control(
                state_trajs[:, :, k, :],
                permuted_controls[:, :, k, :],
            )

            if permuted_disturbances is not None:
                permuted_disturbances[:, :, k, :] = self.dynamics_.clamp_disturbance(
                    state_trajs[:, :, k, :],
                    permuted_disturbances[:, :, k, :],
                )

            state_trajs[:, :, k + 1, :] = self.get_next_step_state(
                state_trajs[:, :, k, :],
                permuted_controls[:, :, k, :],
                None if permuted_disturbances is None else permuted_disturbances[:, :, k, :],
            )

        return state_trajs, permuted_controls, permuted_disturbances

    # def init_control_tensors(self):
    #     self.receiding_start = 0
    #     self.control_init = self.dynamics_.control_init.unsqueeze(
    #         0).repeat(self.batch_size, 1)
    #     self.control_tensors = self.control_init.unsqueeze(
    #         1).repeat(1, self.horizon, 1)  # A * H * D_u

    def init_control_tensors(self):
        self.receiding_start = 0

        self.control_init = self.dynamics_.control_init.unsqueeze(
            0).repeat(self.batch_size, 1)
        self.control_tensors = self.control_init.unsqueeze(
            1).repeat(1, self.horizon, 1)

        if self.dynamics_.disturbance_dim > 0:
            if hasattr(self.dynamics_, "disturbance_init"):
                disturbance_init = self.dynamics_.disturbance_init
            else:
                disturbance_init = torch.zeros(
                    self.dynamics_.disturbance_dim,
                    device=self.device,
                )

            self.disturbance_init = disturbance_init.unsqueeze(
                0).repeat(self.batch_size, 1)
            self.disturbance_tensors = self.disturbance_init.unsqueeze(
                1).repeat(1, self.horizon, 1)
        else:
            self.disturbance_tensors = None


    # def get_next_step_state(self, state, controls):
    #     current_dsdt = self.dynamics_.dsdt(
    #         state, controls, None)
    #     next_states = self.dynamics_.equivalent_wrapped_state(
    #         state + current_dsdt*self.dT)
    #     # next_states = torch.clamp(next_states, self.dynamics_.state_range_[..., 0], self.dynamics_.state_range_[..., 1])
    #     return next_states

    def get_next_step_state(self, state, controls, disturbances=None):
        current_dsdt = self.dynamics_.dsdt(
            state, controls, disturbances)
        next_states = self.dynamics_.equivalent_wrapped_state(
            state + current_dsdt*self.dT)
        return next_states


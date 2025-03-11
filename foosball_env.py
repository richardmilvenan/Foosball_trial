# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.foosball import FOOSBALL_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class FoosballEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0  #We can adjust length as needed
    possible_agents = ["protagonist", "antagonist"]
    action_spaces = {"protagonist": 8, "antagonist": 8}
    observation_spaces = {"protagonist": 38, "antagonist": 38}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = FOOSBALL_CFG.replace(prim_path="/World/envs/env_.*/Foosball")
    
    actuated_joint_names = [
        "Keeper_W_PrismaticJoint",
        "Defense_W_PrismaticJoint",
        "Mid_W_PrismaticJoint",
        "Offense_W_PrismaticJoint",
        "Keeper_B_PrismaticJoint",
        "Defense_B_PrismaticJoint",
        "Mid_B_PrismaticJoint",
        "Offense_B_PrismaticJoint",
        "Keeper_W_RevoluteJoint",
        "Defense_W_RevoluteJoint",
        "Mid_W_RevoluteJoint",
        "Offense_W_RevoluteJoint",
        "Keeper_B_RevoluteJoint",
        "Defense_B_RevoluteJoint",
        "Mid_B_RevoluteJoint",
        "Offense_B_RevoluteJoint",
    ]

    # in-game Ball
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.01725,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 0.75, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2,dynamic_friction=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=10.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.79025), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # reset
    #max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    #initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]
    #initial_pendulum_angle_range = [-0.25, 0.25]  # the range in which the pendulum angle is sampled from on reset [rad]

    # action scales
    #cart_action_scale = 100.0  # [N]
    #pendulum_action_scale = 50.0  # [Nm]

    # reward scales
    #rew_scale_alive = 1.0
    #rew_scale_terminated = -2.0
    #rew_scale_cart_pos = 0
    #rew_scale_cart_vel = -0.01
    #rew_scale_pole_pos = -1.0
    #rew_scale_pole_vel = -0.01
    #rew_scale_pendulum_pos = -1.0
    #rew_scale_pendulum_vel = -0.01


class FoosballEnv(DirectMARLEnv):
    cfg: FoosballEnvCfg

    def __init__(self, cfg: FoosballEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def get_game_ball(self) -> None:
        ball_path = self.default_zero_env_path + "/Foosball/Ball"
        self._init_ball_position = torch.tensor(
            [[0, 0, 0.79025]], device=self.device
        ).repeat(self.num_envs, 1)
        self._init_ball_rotation = torch.tensor(
            [[1, 0, 0, 0]], device=self.device
        ).repeat(self.num_envs, 1)
        self._init_ball_velocities = torch.zeros((self.num_envs, 6), device=self.device)
        self._ball_radius = 0.01725

        self._sim_config.apply_articulation_settings(
            "Ball", get_prim_at_path(ball_path),
            self._sim_config.parse_actor_config("Ball")
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add Table to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        #add ball
        self.get_game_ball()


    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions["cart"] * self.cfg.cart_action_scale, joint_ids=self._cart_dof_idx
        )
        self.robot.set_joint_effort_target(
            self.actions["pendulum"] * self.cfg.pendulum_action_scale, joint_ids=self._pendulum_dof_idx
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pole_joint_pos = normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1))
        pendulum_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1))
        observations = {
            "cart": torch.cat(
                (
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    pole_joint_pos,
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
            "pendulum": torch.cat(
                (
                    pole_joint_pos + pendulum_joint_pos,
                    pendulum_joint_pos,
                    self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_pendulum_pos,
            self.cfg.rew_scale_pendulum_vel,
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]),
            self.joint_vel[:, self._pole_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]),
            self.joint_vel[:, self._pendulum_dof_idx[0]],
            math.prod(self.terminated_dict.values()),
        )
        return total_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            self.cfg.initial_pendulum_angle_range[0] * math.pi,
            self.cfg.initial_pendulum_angle_range[1] * math.pi,
            joint_pos[:, self._pendulum_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.sum(
        torch.square(pole_pos + pendulum_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.sum(torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1)

    total_reward = {
        "cart": rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel,
        "pendulum": rew_alive + rew_termination + rew_pendulum_pos + rew_pendulum_vel,
    }
    return total_reward

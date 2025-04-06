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
from isaaclab.assets import Articulation, ArticulationCfg,RigidObjectCfg,RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg,PhysxCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg



@configclass
class FoosballEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 25.0
    action_space = 8
    observation_space = 41
    state_space = 0

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
    white_joint_names = [
        "Keeper_W_PrismaticJoint",
        "Defense_W_PrismaticJoint",
        "Mid_W_PrismaticJoint",
        "Offense_W_PrismaticJoint",
        "Keeper_W_RevoluteJoint",
        "Defense_W_RevoluteJoint",
        "Mid_W_RevoluteJoint",
        "Offense_W_RevoluteJoint",
    ]

    # in-game Ball
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.018,  #0.01725,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3, env_spacing=4.0, replicate_physics=True)


class FoosballEnv(DirectRLEnv):
    cfg: FoosballEnvCfg

    def __init__(self, cfg: FoosballEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        white_joint_names = [
            "Keeper_W_PrismaticJoint",
            "Defense_W_PrismaticJoint",
            "Mid_W_PrismaticJoint",
            "Offense_W_PrismaticJoint",
            "Keeper_W_RevoluteJoint",
            "Defense_W_RevoluteJoint",
            "Mid_W_RevoluteJoint",
            "Offense_W_RevoluteJoint",
        ]
        

        self.white_dof_indices = list()
        for joint_name in white_joint_names:
            self.white_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.white_dof_indices.sort()
    


        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_velocities = self.object.data.root_vel_w

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        #Add Ball to Scene
        self.object = RigidObject(self.cfg.object_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add Table to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions =  actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions, joint_ids=self.white_dof_indices
        )
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.object_pos,
                self.object_velocities,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.object_pos,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_velocities = self.object.data.root_vel_w
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        out_of_bounds = white_goal(self.object_pos) | black_goal(self.object_pos)
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        
        
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)




@torch.jit.script
def white_goal(object_pos: torch.Tensor) -> torch.Tensor:
    return object_pos[:, 0] > 0.61725 


@torch.jit.script
def black_goal(object_pos: torch.Tensor) -> torch.Tensor:
    return object_pos[:, 0] < -0.61725


@torch.jit.script
def compute_rewards(
    object_pos: torch.Tensor,
    
):
    device = object_pos.device
    score = torch.zeros(object_pos.shape[0], dtype=torch.float32, device=device)
    
    # Check if white team scored a goal
    score[white_goal(object_pos)] = 10000

    # Check if black team scored a goal
    score[black_goal(object_pos)] = -10000

    



    z = torch.zeros_like(object_pos[:, 1])
    y_dist = torch.pow(torch.max(torch.abs(object_pos[:, 1]) - 0.08525, z), 2)
    x_dist_to_goal_white= torch.pow(object_pos[:, 0] - 0.61725, 2)
    dist_to_goal_white= torch.sqrt(x_dist_to_goal_white + y_dist)

    dist_goal=dist_to_goal_white*10
 
    total_reward = score-dist_to_goal_white
    return total_reward

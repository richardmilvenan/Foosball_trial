# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from numpy import random

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

##
# Pre-defined configs
##
from isaaclab_assets import FOOSBALL_CFG  # isort:skip


@configclass
class FoosballSceneCfg(InteractiveSceneCfg):
    """Configuration for a foosball scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    foosball: ArticulationCfg = FOOSBALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    
    # in-game Ball
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.018,   #0.01725,
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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["foosball"]
    ball = scene["object_cfg"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    rounds = 0

    # Obtain indices for white and black foosmen
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
    black_joint_names = [
        "Keeper_B_PrismaticJoint",
        "Defense_B_PrismaticJoint",
        "Mid_B_PrismaticJoint",
        "Offense_B_PrismaticJoint",
        "Keeper_B_RevoluteJoint",
        "Defense_B_RevoluteJoint",
        "Mid_B_RevoluteJoint",
        "Offense_B_RevoluteJoint",
    ]
    
   
    robot.white_dof_indices = list()
    for joint_name in white_joint_names:
        robot.white_dof_indices.append(robot.joint_names.index(joint_name))
    robot.white_dof_indices.sort()
    


    robot.black_dof_indices = list()
    for joint_name in black_joint_names:
        robot.black_dof_indices.append(robot.joint_names.index(joint_name))
    robot.black_dof_indices.sort()
    

    

    # Simulation loop
    while simulation_app.is_running():
        
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            rounds += 1
            print(rounds)
            if rounds % 5==0:
                #ball reset
                rounds=0
                root_state = ball.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                ball.write_root_pose_to_sim(root_state[:, :7])
                ball.write_root_velocity_to_sim(root_state[:, 7:])
                scene.reset()
                print("[INFO]: Resetting Ball state...")
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            #print(robot.data.joint_names)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

        


            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        #efforts = torch.randn_like(robot.data.joint_pos) * 1.0
        whiteeffort=(random.randint(20)-10)*0.3
        blackeffort=(random.randint(20)-10)*0.2
        #print(effort)
        # -- apply action to the robot
        robot.set_joint_effort_target(whiteeffort, joint_ids=robot.white_dof_indices)
        robot.set_joint_effort_target(blackeffort, joint_ids=robot.black_dof_indices)
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = FoosballSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

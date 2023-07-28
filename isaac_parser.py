# %%
import os
import sys
from matplotlib import pyplot as plt
BASEDIR = os.path.dirname(os.path.abspath(__file__))

# %%
from pprint import pprint
import numpy as np
from isaacgym import gymutil, gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from torch import tensor, Tensor
import torch
from typing import List

from config_ import MinimalCfg as Cfg

# %%
'''
from params_proto import PrefixProto, ParamsProto

class Cfg(PrefixProto, cli=False):
    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    
    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5  # [rad]
        }

    class env(PrefixProto, cli=False):
        num_envs = 9
        device = 'cuda:0'
        envSpacing = 1.0
    class asset(PrefixProto, cli=False):
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        fix_base_link = False  # fixe the base of the robot
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        disable_gravity = False        
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

        # name = 'canine'
        # file = '{ROOT_DIR}/resources/robots/canine/mjcf/canineV4_with_head.xml'.format(ROOT_DIR=BASEDIR)

        # name = 'go1'
        # file = '{ROOT_DIR}/resources/robots/go1/xml/go1.xml'.format(ROOT_DIR=BASEDIR)


        foot_name = "foot"  # name of the feet bodies, used to index body state and contact force tensors

        flip_visual_attachments = False
        fix_base_link = False


    
    class terrain(PrefixProto, cli=False):
        static_friction = 1.0
        dynamic_friction = 1.0


    class render(PrefixProto, cli=False):
        enable_viewer_sync = True
        sync_frame_time = True

cfg = Cfg()
'''

# %%
class IsaacParserClass(object):
    """
        Isaac Parser class
    """
    def __init__(self, cfg:Cfg, USE_ISAAC_VIEWER, VERBOSE=True):
        """
            Initalize Issac parser
        """
        self.cfg = cfg
        self.name = cfg.asset.name
        self.rel_xml_path = cfg.asset.file
        self.viewer = None

        self.VERBOSE = VERBOSE
        # Constants
        self.tick        = 0
        self.render_tick = 0
        # Parse an xml file
        if self.rel_xml_path is not None:
            self._parse_xml(cfg)
        # Viewer
        self.USE_ISAAC_VIEWER = USE_ISAAC_VIEWER
        if self.USE_ISAAC_VIEWER:
            self._init_viewer()
        # Reset
        # self.reset()
        # Print
        if self.VERBOSE:
            self.print_info()

    def print_info(self):
        pprint(vars(self.info))

    def _parse_xml(self, cfg:Cfg):
        # Get Empty Simulation Parameters
        self.sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), self.sim_params)

        # Get Empty Gym
        self.gym = gymapi.acquire_gym()

        # Set the physics engine
        self.physics_engine = gymapi.SIM_PHYSX

        # set device
        self.device = cfg.env.device
        self.device_type, self.device_id = gymutil.parse_device_str(self.device)
        graphics_device_id = self.device_id

        asset_path = self.rel_xml_path
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.sim = self.gym.create_sim(self.device_id, graphics_device_id, self.physics_engine, self.sim_params)


        # add ground plane


        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        self.gym.add_ground(self.sim, plane_params)


        # add robot
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode        = cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints         = cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments       = cfg.asset.flip_visual_attachments
        asset_options.fix_base_link                 = cfg.asset.fix_base_link
        asset_options.density                       = cfg.asset.density
        asset_options.angular_damping               = cfg.asset.angular_damping
        asset_options.linear_damping                = cfg.asset.linear_damping
        asset_options.max_angular_velocity          = cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity           = cfg.asset.max_linear_velocity
        asset_options.armature                      = cfg.asset.armature
        asset_options.thickness                     = cfg.asset.thickness
        asset_options.disable_gravity               = cfg.asset.disable_gravity

        assert os.path.isfile(asset_path)
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        class Info(object):
            num_dof: int
            num_bodies: int
            dof_props_asset: np.ndarray
            rigid_shape_props_asset: List
            body_names: List
            joint_names: List
            feet_names: List
            dof_names: List
            base_init_state_list: List
            dof_init_state: List

        self.info = Info()
        self.info.num_dof                 = self.gym.get_asset_dof_count(robot_asset)
        self.info.num_bodies              = self.gym.get_asset_rigid_body_count(robot_asset)
        self.info.dof_props_asset         = self.gym.get_asset_dof_properties(robot_asset)
        self.info.rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.info.body_names              = self.gym.get_asset_rigid_body_names(robot_asset)
        self.info.joint_names             = self.gym.get_asset_dof_names(robot_asset)
        self.info.feet_names              = [s for s in self.info.body_names if cfg.asset.foot_name in s]
        self.info.dof_names               = self.gym.get_asset_dof_names(robot_asset)
        self.info.base_init_state_list    = cfg.init_state.pos + cfg.init_state.rot + cfg.init_state.lin_vel + cfg.init_state.ang_vel
        
        # Create the env
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.info.base_init_state_list[:3])
        breakpoint()
        actor_handles = []
        envs = []

        envSpacing_lower = gymapi.Vec3(-cfg.env.envSpacing, -cfg.env.envSpacing, 0.)
        envSpacing_upper = gymapi.Vec3(cfg.env.envSpacing,cfg.env.envSpacing,cfg.env.envSpacing)


        for i in range(cfg.env.num_envs):
            env_handle = self.gym.create_env(self.sim, envSpacing_lower, envSpacing_upper, int(np.sqrt(cfg.env.num_envs)))
            
            # TODO: randomize [friction, restitution]
            rigid_shape_props = self.info.rigid_shape_props_asset
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, cfg.asset.name, i,
                                                      cfg.asset.disable_self_collisions, 0)

            # TODO: randomize [motor_strength, motor_offset, Kp_factor, Kd_factor]
            dof_props = self.info.dof_props_asset
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            # TODO: randomize [com, mass]
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            envs.append(env_handle)
            actor_handles.append(actor_handle)
        
        self.envs = envs
        self.actor_handles = actor_handles

        # Allocate static memories
        self.gym.prepare_sim(self.sim)

        
        # dof_init_state = self.get_dof_state().clone()
        # dof_init_state[:,:,0] = to_torch(list(self.cfg.init_state.default_joint_angles.values())).to(self.device)
        # dof_init_state[:,:,1] = 0

        # self.set_dof_state(dof_init_state)

        # Set initial state
        # self.info.dof_init_state = dof_init_state.tolist()

        self.foot_indices = []
        for i in range(len(self.info.feet_names)):
            self.foot_indices.append(self.name2index(self.info.feet_names[i]))
    
        self.update()
        self.tick = 0

    def get_body_props(self, env_ids=None):
        '''
        com and mass
        '''
        if env_ids is None:
            env_ids = torch.arange(self.cfg.env.num_envs)
        env_ids_int32 = to_torch(env_ids,dtype=torch.int32,device=self.device)

        body_prop = []
        for env_id in env_ids_int32:
            body_prop_ = self.gym.get_actor_rigid_body_properties(
                self.envs[env_id], self.actor_handles[env_id])
            body_prop.append(body_prop_)
        return body_prop

    def _init_viewer(self):
        if self.viewer is None:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        self.cameras = {}
    
    def update_viewer(self, camera_pos, camera_lookat):
        self.gym.viewer_camera_look_at(
            self.viewer, self.envs[0], gymapi(*camera_pos), gymapi(*camera_lookat)
        )

    def set_camera(self, camera_name, camera_pos, camera_lookat, width=360, height=240):
        if camera_name not in self.cameras.keys():
            camera_props = gymapi.CameraProperties()
            camera_props.width = width
            camera_props.height = height
            camera_ = self.gym.create_camera_sensor(self.envs[0], camera_props)
        else:
            camera_ = self.cameras[camera_name]

        self.gym.set_camera_location(camera_, self.envs[0], gymapi.Vec3(*camera_pos), gymapi.Vec3(*camera_lookat))
        self.cameras[camera_name] = camera_
        return camera_

    def grab_image(self, camera_):
        if isinstance(camera_,str):
            camera_ = self.cameras[camera_]
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], camera_, gymapi.IMAGE_COLOR)
        w,h = img.shape
        img = img.reshape([w,h//4,4])
        return img
    

    def update(self):
        # Update states
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
    
    def get_root_state(self):
        '''
        0~3: position
        3~7: quaternion
        7~10: linear velocity
        10~13: angular velocity
        '''
        return gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim))\
                .view(self.cfg.env.num_envs, 13)\

    def get_body_state(self):
        '''
        0~3: position
        3~7: quaternion
        7~10: linear velocity
        10~13: angular velocity
        '''
        return gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim))\
                .view(self.cfg.env.num_envs, self.info.num_bodies, 13)\

    def get_dof_state(self):
        '''
        0: joint values
        1: joint velocities
        '''
        return gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim))\
                .view(self.cfg.env.num_envs, self.info.num_dof, 2)\

    def get_contact_force(self):
        return gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim))\
                .view(self.cfg.env.num_envs, self.info.num_bodies, 3)\
    
    def get_dof_force(self):
        return gymtorch.wrap_tensor(
            self.gym.acquire_dof_force_tensor(self.sim))\
                .view(self.cfg.env.num_envs, self.info.num_dof)\


    def set_root_state(self, root_state:Tensor, env_ids:Tensor=None):
        '''
        root_state:[num_env, 13]
        '''
        if env_ids is None:
            env_ids = torch.arange(self.cfg.env.num_envs)
        env_ids_int32 = to_torch(env_ids,dtype=torch.int32,device=self.device)
        root_state_ = self.get_root_state()
        root_state_[env_ids_int32] = root_state

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_state_),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def set_body_state(self):
        raise NotImplementedError("Issac Gym have unstable behavior when setting body state. Thus forbid using it")

    def set_dof_state(self, dof_state:Tensor, env_ids:Tensor=None):
        '''
        dof_state: [num_env, num_dof x 2]
        joint_values: [:,:,0]
        joint_velocities: [:,:,1]
        '''
        if env_ids is None:
            env_ids = torch.arange(self.cfg.env.num_envs)
        env_ids_int32 = to_torch(env_ids,dtype=torch.int32,device=self.device)

        dof_state_ = self.get_dof_state().view(self.cfg.env.num_envs, -1)
        dof_state_[env_ids_int32] = dof_state.view(self.cfg.env.num_envs, -1)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(dof_state_),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def set_contact_force(self):
        raise NotImplementedError("Can not set contact force")

    def set_dof_force(self, dof_force:Tensor, env_ids:Tensor=None):
        '''
        dof_force: [num_env, num_dof]
        '''
        if env_ids is None:
            env_ids = torch.arange(self.cfg.env.num_envs)
        env_ids_int32 = to_torch(env_ids,dtype=torch.int32,device=self.device)

        dof_force_ = self.get_dof_force()
        dof_force_[env_ids_int32] = dof_force
        self.gym.set_dof_actuation_force_tensor(
            self.sim,
              gymtorch.unwrap_tensor(dof_force))

    def body2index(self, name):
        return self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], name, gymapi.DOMAIN_ACTOR)

    def joint2index(self, name):
        return self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], name)

    def name2index(self, name):
        if name in self.info.body_names:
            return self.body2index(name)
        elif name in self.info.joint_names:
            return self.joint2index(name)
        else:
            raise ValueError("Unknown name: {}\
                             Refer to self.cfg.info.body_names or cfg.info_joint_names".format(name))

    def names2indices(self, names):
        indices = []
        for name in names:
            indices.append(self.name2index(name))
        return indices


    def step(self, torques):
        assert torques.shape == (self.cfg.env.num_envs, self.info.num_dof)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

        # step
        self.gym.simulate(self.sim)
        self.update()
        self.render()
    
    def render(self):
        enable_viewer_sync = True

        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                enable_viewer_sync = not enable_viewer_sync

        # step graphics
        if enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            # if self.cfg.render.sync_frame_time:
            self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer) 
    
    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def reset(self):
        root_state = to_torch(self.info.base_init_state_list)
        root_state = root_state.reshape(1,-1).repeat(self.cfg.env.num_envs,1)
        self.set_root_state(root_state)
        
        self.set_dof_force(torch.zeros(self.cfg.env.num_envs, self.info.num_dof).to(self.device))

        # self.gym.simulate(self.sim)
        self.update()


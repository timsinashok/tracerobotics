"""Inline MJCF XML models for task environments.

Keeps all model definitions self-contained — no external asset files needed.
"""

PANDA_7DOF_REACH_XML: str = """
<mujoco model="panda_reach">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <joint damping="1.0" armature="0.1"/>
    <geom condim="3" friction="1.0 0.005 0.0001" rgba="0.7 0.7 0.7 1"/>
    <position kp="100" kv="10"/>
  </default>

  <worldbody>
    <!-- Table surface -->
    <geom name="table" type="box" size="0.6 0.6 0.02" pos="0.4 0 0.0"
          rgba="0.4 0.3 0.2 1" contype="1" conaffinity="1"/>

    <!-- 7-DOF arm (Panda-like kinematics) mounted on table -->
    <body name="link0" pos="0 0 0.02">
      <geom type="cylinder" size="0.06 0.05" rgba="0.9 0.9 0.9 1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.04"/>
        <body name="link2" pos="0 0 0.15">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.7628 1.7628"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.04"/>
          <body name="link3" pos="0 0 0.15">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.035"/>
            <body name="link4" pos="0 0 0.12">
              <joint name="joint4" type="hinge" axis="0 1 0" range="-3.0718 -0.0698"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.035"/>
              <body name="link5" pos="0 0 0.12">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.03"/>
                <body name="link6" pos="0 0 0.1">
                  <joint name="joint6" type="hinge" axis="0 1 0" range="-0.0175 3.7525"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.025"/>
                  <body name="link7" pos="0 0 0.08">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="sphere" size="0.03" rgba="0.2 0.2 0.8 1"/>
                    <site name="end_effector" pos="0 0 0.04" size="0.01"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Mocap target (green sphere, non-colliding) -->
    <body name="target" mocap="true" pos="0.4 0.0 0.3">
      <geom name="target_geom" type="sphere" size="0.02"
            rgba="0.2 0.9 0.2 0.7" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="act1" joint="joint1" kp="100" kv="10" ctrlrange="-2.8973 2.8973"/>
    <position name="act2" joint="joint2" kp="100" kv="10" ctrlrange="-1.7628 1.7628"/>
    <position name="act3" joint="joint3" kp="100" kv="10" ctrlrange="-2.8973 2.8973"/>
    <position name="act4" joint="joint4" kp="100" kv="10" ctrlrange="-3.0718 -0.0698"/>
    <position name="act5" joint="joint5" kp="100" kv="10" ctrlrange="-2.8973 2.8973"/>
    <position name="act6" joint="joint6" kp="100" kv="10" ctrlrange="-0.0175 3.7525"/>
    <position name="act7" joint="joint7" kp="100" kv="10" ctrlrange="-2.8973 2.8973"/>
  </actuator>
</mujoco>
"""

import mujoco

# urdf_file = "./asset/103486/textured_objs/mobility_vhacd.urdf"
urdf_file = "./asset/ibm/mesh_temp/panda_backup.urdf"
# urdf_file = "./test.urdf"

mujoco.MjModel.from_xml_path(urdf_file)
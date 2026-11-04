"""Armature receive: a posed mesh takes edits from Nomad as inverse-skinned rest positions.

Runs inside Blender: blender --background --factory-startup --python tests/test_armature_receive.py
"""
import os, sys
import numpy
import bpy
from mathutils import Matrix

ADDON_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ADDON_DIR)
import nomad_blender_link as addon
addon.register()

fails = []
addon.remote_capabilities = {"ngon"}


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label} {detail}")
    if not condition:
        fails.append(label)


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, depth=3.0, radius=0.5, location=(0.3, -0.2, 1.5))
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=6)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.armature_add(location=(0.3, -0.2, 0.0))
    arm = bpy.context.object
    arm.rotation_euler = (0.0, 0.0, 0.4)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = arm.data.edit_bones
    bones[0].name = "a"
    bones[0].head, bones[0].tail = (0, 0, 0), (0, 0, 1)
    parents = {"b": "a", "c": "b"}
    for name, head, tail in (("b", (0, 0, 1), (0, 0, 2)), ("c", (0, 0, 2), (0, 0, 3))):
        bone = bones.new(name)
        bone.head, bone.tail = head, tail
        bone.parent = bones[parents[name]]
        bone.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    for name in "abc":
        obj.vertex_groups.new(name=name)
    to_armature = arm.matrix_world.inverted() @ obj.matrix_world
    for vertex in obj.data.vertices:
        z = min(max((to_armature @ vertex.co).z, 0.0), 3.0)
        low = min(int(z), 2)
        t = z - low
        obj.vertex_groups[low].add([vertex.index], 1.0 - t if low < 2 else 1.0, "REPLACE")
        if t > 0 and low < 2:
            obj.vertex_groups[low + 1].add([vertex.index], t, "REPLACE")
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in arm.pose.bones:
        pose_bone.rotation_mode = "XYZ"
    arm.pose.bones["a"].rotation_euler = (0.3, 0.0, 0.0)
    arm.pose.bones["b"].rotation_euler = (0.0, 0.7, 0.0)
    arm.pose.bones["c"].rotation_euler = (0.5, 0.0, 0.4)
    arm.pose.bones["c"].scale = (1.0, 1.2, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.scene.nomad_link_send_modifiers = True
    bpy.context.view_layer.update()
    return obj, arm


def posed(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    values = addon.foreach_get(mesh.vertices, "co", 3).astype(numpy.float64)
    evaluated.to_mesh_clear()
    return values


def sent(obj):
    header, binary = addon.encode_object(obj)
    return header, bytearray(binary)


def set_positions(header, binary, values):
    binary[: header["vertex_count"] * 12] = addon.to_nomad_vectors(values).astype("<f4").tobytes()
    header["binary_size"] = len(binary)


def delta_packet(obj, indices, values):
    binary = bytearray(numpy.asarray(indices, "<u4").tobytes())
    position_offset = len(binary)
    binary += addon.to_nomad_vectors(values).astype("<f4").tobytes()
    header = {"type": "mesh_delta", "mesh_id": addon.ensure_link_id(obj), "count": len(indices), "index_offset": 0,
              "position_offset": position_offset, "binary_size": len(binary)}
    return header, binary


def rest_state(obj):
    return addon.foreach_get(obj.data.vertices, "co", 3).astype(numpy.float64)


def groups_state(obj):
    return [(g.name, [(v.index, round(e.weight, 5)) for v in obj.data.vertices for e in v.groups if e.group == g.index]) for g in obj.vertex_groups]


def receive(function, header, binary):
    addon.connection.error = ""
    try:
        function(header, bytes(binary))
    except RuntimeError as exc:
        return str(exc)
    bpy.context.view_layer.update()
    return addon.connection.error


offset = numpy.array([0.05, -0.03, 0.04])

obj, arm = build()
header, binary = sent(obj)
before = posed(obj)
rest = rest_state(obj)
groups = groups_state(obj)
check("pose moves the mesh", numpy.abs(before - rest).max() > 0.3)
check("gate passes", addon.armature_gate(obj, len(obj.data.vertices), len(obj.data.polygons)))
sculpted = before + offset
set_positions(header, binary, sculpted)
error = receive(addon.receive_mesh, header, binary)
after = posed(obj)
check("full receive: no error", error == "", error)
check("full receive: evaluated equals sent posed", numpy.abs(after - sculpted).max() < 1e-4, f"{numpy.abs(after - sculpted).max():.2e}")
check("full receive: rest differs from posed", numpy.abs(rest_state(obj) - sculpted).max() > 0.1)
check("full receive: vertex groups kept", groups_state(obj) == groups)
check("full receive: armature modifier kept", len(obj.modifiers) == 1 and obj.modifiers[0].type == "ARMATURE")

obj, arm = build()
before = posed(obj)
moved = numpy.arange(0, len(before), 7)
targets = before[moved] + offset
packet, data = delta_packet(obj, moved, targets)
error = receive(addon.receive_delta, packet, data)
after = posed(obj)
others = numpy.setdiff1d(numpy.arange(len(before)), moved)
check("delta receive: no error", error == "", error)
check("delta receive: moved vertices hit target", numpy.abs(after[moved] - targets).max() < 1e-4, f"{numpy.abs(after[moved] - targets).max():.2e}")
check("delta receive: other vertices untouched", numpy.abs(after[others] - before[others]).max() < 1e-4)

obj, arm = build()
header, binary = sent(obj)
set_positions(header, binary, posed(obj) + offset)
obj.modifiers.new("Extra", "SIMPLE_DEFORM")
rest = rest_state(obj)
error = receive(addon.receive_mesh, header, binary)
check("two modifiers: refused", error == addon.MODIFIER_MESSAGE)
check("two modifiers: mesh untouched", numpy.array_equal(rest, rest_state(obj)))

obj, arm = build()
obj.modifiers[0].use_deform_preserve_volume = True
header, binary = sent(obj)
check("preserve volume: refused", receive(addon.receive_mesh, header, binary) == addon.MODIFIER_MESSAGE)

obj, arm = build()
header, binary = sent(obj)
header["vertex_count"] += 1
check("vertex count change: refused", receive(addon.receive_mesh, header, binary) == addon.MODIFIER_MESSAGE)

obj, arm = build()
header, binary = sent(obj)
header["layers"] = [{"name": "L"}]
check("layers: refused", receive(addon.receive_mesh, header, binary) == addon.MODIFIER_MESSAGE)

obj, arm = build()
obj.modifiers[0].use_deform_preserve_volume = True
packet, data = delta_packet(obj, [0], numpy.zeros((1, 3)))
check("delta gate failure: refused", receive(addon.receive_delta, packet, data) == addon.MODIFIER_MESSAGE)

for label, change in (
    ("modifier viewport off", lambda o: setattr(o.modifiers[0], "show_viewport", False)),
    ("mask vertex group", lambda o: setattr(o.modifiers[0], "vertex_group", "a")),
    ("rest position", lambda o: setattr(o.modifiers[0].object.data, "pose_position", "REST")),
    ("deform bbone", lambda o: setattr(o.modifiers[0].object.data.bones["b"], "bbone_segments", 3)),
    ("vertex groups off", lambda o: setattr(o.modifiers[0], "use_vertex_groups", False)),
    ("bone envelopes on", lambda o: setattr(o.modifiers[0], "use_bone_envelopes", True)),
    ("shape keys", lambda o: o.shape_key_add(name="Basis")),
):
    obj, arm = build()
    header, binary = sent(obj)
    change(obj)
    check(f"{label}: refused", receive(addon.receive_mesh, header, binary) == addon.MODIFIER_MESSAGE)

obj, arm = build()
header, binary = sent(obj)
header["corner_count"] -= int(addon.read_array(binary, header["face_size_offset"], header["face_count"], "<i4")[-1])
header["face_count"] -= 1
check("face count change: refused", receive(addon.receive_mesh, header, binary) == addon.MODIFIER_MESSAGE)

obj, arm = build()
header, binary = sent(obj)
set_positions(header, binary, posed(obj) + offset)
check("first receive", receive(addon.receive_mesh, header, binary) == "")
for vertex in obj.data.vertices:
    if vertex.index % 3 == 0:
        obj.vertex_groups["a"].add([vertex.index], 0.2, "REPLACE")
        obj.vertex_groups["c"].add([vertex.index], 0.8, "REPLACE")
bpy.context.view_layer.update()
header, binary = sent(obj)
sculpted = posed(obj) + offset
set_positions(header, binary, sculpted)
check("second receive after weight edit", receive(addon.receive_mesh, header, binary) == "")
check("weight edit: evaluated equals sent posed", numpy.abs(posed(obj) - sculpted).max() < 1e-4, f"{numpy.abs(posed(obj) - sculpted).max():.2e}")

obj, arm = build()
header, binary = sent(obj)
sculpted = posed(obj) + offset
set_positions(header, binary, sculpted)
world = Matrix.Translation((1.0, 0.5, 0.25)) @ Matrix.Rotation(0.5, 4, "Z") @ obj.matrix_world
header["world_matrix"] = addon.matrix_to_columns(addon.TO_NOMAD @ world @ addon.TO_BLENDER)
check("world matrix change: no error", receive(addon.receive_mesh, header, binary) == "")
check("world matrix change: object moved", numpy.abs(numpy.array(obj.matrix_world) - numpy.array(world)).max() < 1e-4)
check("world matrix change: evaluated equals sent posed", numpy.abs(posed(obj) - sculpted).max() < 1e-4, f"{numpy.abs(posed(obj) - sculpted).max():.2e}")

obj, arm = build()
obj.modifiers[0].use_deform_preserve_volume = True
header, binary = sent(obj)
try:
    addon.receive_packet(header, bytes(binary))
    surfaced = ""
except Exception as exc:
    surfaced = str(exc)
check("receive_packet: refusal raises for the dispatch loop", surfaced == addon.MODIFIER_MESSAGE, surfaced)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)

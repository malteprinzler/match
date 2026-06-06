load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

pytype_strict_library(
    name = "utils",
    srcs = ["__init__.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/models:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/models/gs_render:__pkg__",
    ],
    deps = [
        ":geo_util",
        ":op_util",
        ":util",
        ":vis_util",
    ],
)

pytype_strict_library(
    name = "vis_util",
    srcs = ["vis_util.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/scripts:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/scripts/snippets:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/models:__pkg__",
    ],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/einops",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pudb",
        "//third_party/py/torch:pytorch",
        "//vr/perception/multiview_mesh_prediction/pytorch/utils:render_helper",
    ],
)

pytype_strict_library(
    name = "f250_utils",
    srcs = ["f250_utils.py"],
    visibility = ["//vr/perception/multiview_mesh_prediction/gtempeh/scripts:__pkg__"],
    deps = [
        "//vr/perception/holobooth/data_management/python:holobooth_take_contents",
        "//vr/perception/ubercapture/data_management/python:f250_data_context",
        "//vr/perception/ubercapture/data_management/python:f250_service_type_helper",
    ],
)

pytype_strict_library(
    name = "file_helper",
    srcs = ["file_helper.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:__pkg__",
    ],
    deps = [
        "//pyglib:gfile",
        "//pyglib/concurrent:parallel",
        "//pyglib/contrib/gpathlib:gpath",
        "//research/vale/tools/image_scrape:remote_file_wrapper_main_lib",
        "//third_party/py/accelerate",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "image_helper",
    srcs = ["image_helper.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/scripts:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/models:__pkg__",
    ],
    deps = [
        ":file_helper",
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/py/cv2",
        "//third_party/py/imageio",
        "//third_party/py/numpy",
        "//third_party/py/skimage",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_addons/image",
        "//third_party/py/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "uv_util",
    srcs = ["uv_util.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/scripts:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data/serialization:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/models:__pkg__",
    ],
    deps = [
        ":image_helper",
        "//third_party/py/einops",
        "//third_party/py/numpy",
        "//third_party/py/pudb",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_graphics/rendering:rasterization_backend",
        "//third_party/py/tensorflow_graphics/rendering:triangle_rasterizer",
        "//third_party/py/torch:pytorch",
        "//vr/perception/multiview_mesh_prediction/tensorflow/utils:holobooth_camera",
        "//vr/perception/multiview_mesh_prediction/tensorflow/utils:render_helper",
    ],
)

pytype_strict_binary(
    name = "uv_util_cli",
    srcs = ["uv_util_cli.py"],
    data = ["//vr/perception/multiview_mesh_prediction/gtempeh:assets"],
    visibility = [
    ],
    deps = [
        ":uv_util",
        "//pyglib:resources",
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/py/absl:app",
        "//third_party/py/einops",
        "//third_party/py/matplotlib:matplotlib_interactive",
        "//third_party/py/pudb",
        "//third_party/py/torch:pytorch",
        "//third_party/py/tqdm",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:options",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:holobooth_dataset",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:serialized_holobooth_dataset",
    ],
)

pytype_strict_library(
    name = "mesh_util",
    srcs = ["mesh_util.py"],
    visibility = ["//vr/perception/multiview_mesh_prediction/gtempeh/src/models:__pkg__"],
    deps = [
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/scripts:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:__pkg__",
    ],
    deps = [
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/py/absl/flags",
        "//third_party/py/accelerate",
        "//third_party/py/einops",
        "//third_party/py/huggingface_hub",  # buildcleaner: keep
        "//third_party/py/omegaconf",
        "//third_party/py/torch:pytorch",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:options",
    ],
)

pytype_strict_library(
    name = "op_util",
    srcs = ["op_util.py"],
    deps = [
        "//third_party/py/einops",
        "//third_party/py/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "geo_util",
    srcs = ["geo_util.py"],
    deps = [
        "//third_party/py/torch:pytorch",
    ],
)

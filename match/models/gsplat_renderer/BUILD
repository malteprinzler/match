load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

pytype_strict_library(
    name = "gsplat_renderer",
    srcs = ["gsplat_renderer.py"],
    visibility = [
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/data:__pkg__",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src/models:__pkg__",
    ],
    deps = [
        "//third_party/gsplat:gsplat/rendering",
        "//third_party/py/einops",
        "//third_party/py/pudb",
        "//third_party/py/torch:pytorch",
        "//vr/perception/multiview_mesh_prediction/gtempeh/src:options",
    ],
)

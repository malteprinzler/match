from setuptools import find_packages, setup

# Vendored deps under third_party/ are not part of the match package.
_EXCLUDE = ("third_party", "third_party.*")


def _match_packages():
    discovered = find_packages(exclude=_EXCLUDE)
    return ["match"] + [f"match.{name}" for name in discovered]


setup(
    name="match",
    version="0.1.0",
    package_dir={"match": "."},
    packages=_match_packages(),
    python_requires=">=3.10",
    install_requires=[],
)

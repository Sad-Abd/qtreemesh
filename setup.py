import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "qtreemesh",
    version = "0.1.01",
    author = "Sad-Abd",
    author_email = "abedisadjad@gmail.com",
    description = "A package that creats quadtree mesh from an image",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Sad-Abd/qtreemesh",
    project_urls = {
        "Bug Tracker": "https://github.com/Sad-Abd/qtreemesh/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requiers = [
        'numpy',
        'matplotlib',
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)
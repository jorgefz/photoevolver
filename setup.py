import setuptools

ext_path = "photoevolver/cmodels/src"
ext_cmodels = setuptools.Extension(
    "photoevolver.cmodels",
    [
        ext_path+"/mloss.c",
        ext_path+"/struct.c",
        ext_path+"/cmodels.c",
    ],
    depends = [
        ext_path+"/constants.h",
        ext_path+"/models.h",
    ],
    include_dirs = [ext_path]   
)

setuptools.setup(ext_modules=[ext_cmodels], include_package_data=True)


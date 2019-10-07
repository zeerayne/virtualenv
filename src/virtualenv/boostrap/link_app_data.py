from appdirs import user_data_dir

WHEEL_CACHE = user_data_dir(appname="virtualenv", appauthor="pypa")


def bootstrap(interpreter):
    packages = interpreter.options.seed_packages
    target = interpreter.site_packages[0]

from pathlib import Path
import logging
import coloredlogs


# create Filehandler that logs to /temp/app.log
log_path = (Path(__file__).parent / 'app.log')
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.DEBUG,
    format=log_fmt,
    filename=log_path,
    filemode='w',
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(log_fmt)
console.setFormatter(fmt=formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

# create logger with different names to make debugging easier
logger_data = logging.getLogger('src.data')
logger_viz = logging.getLogger('src.visualization')
logger_models = logging.getLogger('src.models')
logger_hp_optim = logging.getLogger('src.tuning')

# use colored logs
coloredlogs.install()

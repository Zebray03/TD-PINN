from config import Config
from model import BurgersTransformer
from train import train
from utils import plot_results

# 初始化
config = Config()
model = BurgersTransformer(config).to(config.device)

print("Begin")
losses = train(model, config)

plot_results(losses, [])

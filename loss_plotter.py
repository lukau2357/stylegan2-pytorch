import matplotlib.pyplot as plt
import csv
import os
import argparse
import json

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("model_dir", type = str, help = "Model directory used during training")
    return args.parse_args()

def get_metric(filename):
    with open(filename, "r", encoding = "utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        metrics = [float(item[0]) for item in reader if isinstance(item, list)]
        return metrics

def plot_data(metric, xlabel, ylabel, output_name):
    fig, ax = plt.subplots()
    ax.plot(metric)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(output_name)

def plot_metrics(dir):
    metadata_path = os.path.join(dir, ".metadata.json")
    with open(metadata_path, "r", encoding = "utf-8") as f:
        metadata = json.load(f)
        metadata_pretty = json.dumps(metadata, indent = 4)

    print(f"Found model directory at {dir}. Model metadata:")
    print(metadata_pretty)

    d_adv_loss = get_metric(os.path.join(dir, "d_adversarial_loss.csv"))
    g_adv_loss = get_metric(os.path.join(dir, "g_adversarial_loss.csv"))
    metric_count = 2

    g_plr, d_gp = None, None

    if metadata["use_plr"]:
        g_plr = get_metric(os.path.join(dir, "g_plr.csv"))
        metric_count += 1
    
    if metadata["use_gp"]:
        d_gp = get_metric(os.path.join(dir, "d_gp.csv"))
        metric_count += 1
    
    plt.style.use("ggplot")
    plot_data(d_adv_loss, f"Discriminator Step. Discriminator optim steps: {metadata['disc_optim_steps']}", "Average Discriminator Adversarial Loss", os.path.join(dir, "d_adversarial_loss.png"))
    plot_data(g_adv_loss, "Generator Step", "Generator Adversarial Loss", os.path.join(dir, "g_adversarial_loss.png"))
    
    if g_plr is not None:
        plot_data(g_plr, f"Generator PLR step with period {metadata['lazy_reg_steps_generator']}", "PLR Value", os.path.join(dir, "g_plr.png"))
    
    if d_gp is not None:
        plot_data(d_gp, f"Discriminator GP step with period {metadata['lazy_reg_steps_discriminator']}", "Average GP", os.path.join(dir, "d_gp.png"))
    
if __name__ == "__main__":
    args = parse_args()
    plot_metrics(args.model_dir)
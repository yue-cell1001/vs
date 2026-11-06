import csv
from pathlib import Path

def write_csv(series, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['series', 'iteration', 'hv'])
        for label, history in series:
            writer.writerows((label, row['iteration'], row['hv']) for row in history if 'hv' in row)

def draw(series, destination, xlabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for label, history in series:
        points = [row for row in history if 'hv' in row]
        if not points:
            continue
        ax.plot([row['iteration'] for row in points], [row['hv'] for row in points], label=label)
        rows.extend((label, row['iteration'], row['hv']) for row in points)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Hypervolume (higher is better)')
    ax.grid(alpha=0.25)
    if rows:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No recorded iterations', ha='center', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    with destination.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['series', 'iteration', 'hv'])
        writer.writerows(rows)

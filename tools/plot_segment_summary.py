import json
import os
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT = os.path.join(ROOT, "output", "segment_summary.json")
OUT_PNG = os.path.join(ROOT, "output", "segment_summary.png")


def main():
    if not os.path.exists(INPUT):
        print("No segment_summary.json found at", INPUT)
        return

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No segments to plot")
        return

    idx = [s["segment_index"] for s in segments]
    totals = [s["total_opportunities"] for s in segments]
    whiffs = [s["whiffs"] for s in segments]
    whiff_pun = [s["whiff_punishable"] for s in segments]

    # Plota contagens por segmento. Labels em Português (pt-BR).
    plt.figure(figsize=(10, 4))
    plt.plot(idx, totals, marker='o', label='oportunidades totais')
    plt.plot(idx, whiffs, marker='o', label='whiffs')
    plt.plot(idx, whiff_pun, marker='o', label='whiffs puníveis')
    plt.xlabel('Segmento (índice)')
    plt.ylabel('Contagem')
    plt.title('Oportunidades de punição por segmento')
    plt.legend(title='Legenda')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print('Saved plot to', OUT_PNG)


if __name__ == '__main__':
    main()

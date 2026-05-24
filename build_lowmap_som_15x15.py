#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import pandas as pd
from minisom import MiniSom
import laytr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfeat", required=True, help="laytr kfeat .jl file")
    parser.add_argument("--bed", required=True, help="BED used for kfeat")
    parser.add_argument("--out-prefix", required=True, help="Output prefix")
    parser.add_argument("--som-x", type=int, default=15, help="SOM x dimension")
    parser.add_argument("--som-y", type=int, default=15, help="SOM y dimension")
    parser.add_argument("--iterations", type=int, default=20000, help="Training iterations")
    parser.add_argument("--sigma", type=float, default=2.5, help="MiniSom sigma")
    parser.add_argument("--learning-rate", type=float, default=1.0, help="MiniSom learning rate")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    args = parser.parse_args()

    # Load laytr k-mer features
    kfeats_raw = joblib.load(args.kfeat)
    if "features" not in kfeats_raw:
        raise KeyError("Expected key 'features' in laytr kfeat .jl file")

    kfeats = np.asarray(kfeats_raw["features"])
    print("kfeats shape:", kfeats.shape)

    # Read BED
    bed = pd.read_csv(
        args.bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"]
    )
    print("BED rows:", len(bed))

    if len(bed) != kfeats.shape[0]:
        raise ValueError(
            f"BED rows ({len(bed)}) do not match kfeat rows ({kfeats.shape[0]})."
        )

    # Train SOM
    som = MiniSom(
        args.som_x,
        args.som_y,
        kfeats.shape[1],
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        topology="hexagonal",
        neighborhood_function="gaussian",
        activation_distance="euclidean",
        random_seed=args.seed,
    )

    print("Training SOM...")
    som.train_batch(kfeats, args.iterations, verbose=True)
    print("Quantization error:", som.quantization_error(kfeats))

    # Save SOM
    som_file = f"{args.out_prefix}.som"
    joblib.dump(som, som_file)

    # Map each BED row to a SOM node
    m_map = laytr.map_to_som(kfeats, som)
    print("Mapped rows:", len(m_map))

    if len(m_map) != len(bed):
        raise ValueError(
            f"Mapped rows ({len(m_map)}) do not match BED rows ({len(bed)})."
        )

    # Per-window data table
    data = bed.copy()
    data["x"] = [int(i[0]) for i in m_map]
    data["y"] = [int(i[1]) for i in m_map]

    weights = som.get_weights()
    qerrs = []
    for row_idx, feat in enumerate(kfeats):
        x = data.loc[row_idx, "x"]
        y = data.loc[row_idx, "y"]
        qerrs.append(np.linalg.norm(feat - weights[x, y]))
    data["quantization_error"] = qerrs

    data_file = f"{args.out_prefix}_data.tsv"
    data.to_csv(data_file, sep="\t", index=False)

    # Occupancy tables
    node_counts = (
        data.groupby(["x", "y"])
        .size()
        .reset_index(name="n_windows")
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )
    node_counts.to_csv(f"{args.out_prefix}_node_counts.tsv", sep="\t", index=False)

    all_nodes = pd.MultiIndex.from_product(
        [range(args.som_x), range(args.som_y)],
        names=["x", "y"]
    ).to_frame(index=False)

    node_counts_full = all_nodes.merge(node_counts, on=["x", "y"], how="left")
    node_counts_full["n_windows"] = node_counts_full["n_windows"].fillna(0).astype(int)
    node_counts_full.to_csv(f"{args.out_prefix}_node_counts_full.tsv", sep="\t", index=False)

    # Occupancy matrix
    umatrix = np.zeros(som.get_weights().shape[:2], dtype=int)
    for idx in m_map:
        umatrix[tuple(idx)] += 1
    np.save(f"{args.out_prefix}_occupancy.npy", umatrix)

    # Occupancy plot
    plot = laytr.make_hex_plot(
        som,
        hue=umatrix,
        hue_label="Number of 500 bp windows",
        hue_count_ticks=True
    )
    plot.axes[0].set_title(
        f"{args.som_x}x{args.som_y} SOM occupancy for low-mappability 500 bp windows",
        pad=10
    )
    plot.figure.savefig(f"{args.out_prefix}_occupancy.png", dpi=300, bbox_inches="tight")

    # Summary
    total_nodes = args.som_x * args.som_y
    used_nodes = (node_counts_full["n_windows"] > 0).sum()
    empty_nodes = total_nodes - used_nodes

    print("\n=== SUMMARY ===")
    print("Total nodes:", total_nodes)
    print("Used nodes:", used_nodes)
    print("Empty nodes:", empty_nodes)
    print("Percent empty:", round(empty_nodes / total_nodes * 100, 2))

    print("\nSaved files:")
    print(som_file)
    print(data_file)
    print(f"{args.out_prefix}_node_counts.tsv")
    print(f"{args.out_prefix}_node_counts_full.tsv")
    print(f"{args.out_prefix}_occupancy.npy")
    print(f"{args.out_prefix}_occupancy.png")

if __name__ == "__main__":
    main()

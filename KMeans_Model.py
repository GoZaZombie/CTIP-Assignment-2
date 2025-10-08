import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib

def load_text_df(path):
    df = pd.read_csv(path, low_memory=False)
    if "text" not in df.columns:
        if "v2" in df.columns:
            df = df.rename(columns={"v2": "text"})
        else:
            raise ValueError("CSV must have a 'text' column (or 'v2').")
    df["text"] = df["text"].astype(str)
    cols = ["text"] + ([c for c in ["label"] if c in df.columns])
    return df[cols]

def tfidf_svd(texts, n_components=100):
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=1, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xs = svd.fit_transform(X)
    return Xs, tfidf, svd

def elbow_plot(X, out_png, kmin=2, kmax=10):
    inertias, models = [], {}
    for k in range(kmin, kmax + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_); models[k] = km
    plt.figure(); plt.plot(range(kmin, kmax + 1), inertias, marker="o")
    plt.xlabel("k"); plt.ylabel("Inertia (WCSS)"); plt.title("Elbow Method")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return models

def plot_2d(X, labels, out_png):
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd2.fit_transform(X)
    plt.figure(); plt.scatter(X2[:,0], X2[:,1], c=labels, s=10)
    plt.title("K-Means (2D SVD projection)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="email_dataset.csv")
    p.add_argument("--outdir", default="ModelTraining")
    p.add_argument("--k", type=int, default=3, help="clusters to use (after elbow)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_text_df(args.data)
    print(f"[KMEANS] Loaded {args.data} -> {df.shape}")

    Xs, tfidf, svd = tfidf_svd(df["text"], n_components=100)
    print(f"[KMEANS] SVD shape: {Xs.shape}")

    models = elbow_plot(Xs, os.path.join(args.outdir, "elbow.png"), kmin=2, kmax=10)
    k = args.k
    km = models[k]
    labels = km.labels_
    sil = silhouette_score(Xs, labels)
    print(f"[KMEANS] k={k}  silhouette={sil:.3f}")

    df_out = df.copy(); df_out["cluster"] = labels
    out_csv = os.path.join(args.outdir, f"clustered_{os.path.basename(args.data)}")
    df_out.to_csv(out_csv, index=False)

    joblib.dump(tfidf, os.path.join(args.outdir, "KMeansVectorizer.pkl"))
    joblib.dump(svd,   os.path.join(args.outdir, "KMeansSVD.pkl"))
    joblib.dump(km,    os.path.join(args.outdir, "KMeansModel.pkl"))

    plot_2d(Xs, labels, os.path.join(args.outdir, "clusters_2d.png"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"k": k, "silhouette": float(sil)}, f, indent=2)

    print(f"[KMEANS] Saved: {out_csv} and artifacts in {args.outdir}")

if __name__ == "__main__":
    main()

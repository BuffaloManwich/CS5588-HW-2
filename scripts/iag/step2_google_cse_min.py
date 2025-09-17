import os, json, time, argparse
from collections import defaultdict
from googleapiclient.discovery import build
def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
def main(dataset, max_queries, per_query, sleep):
    base = f"intermediate_files/{dataset}"
    id2all = load_json(f"{base}/id2allquery.json")
    # also merge cluster queries if present
    id2clu = {}
    p_clu = f"{base}/id2clusterquery.json"
    if os.path.exists(p_clu):
        try: id2clu = load_json(p_clu)
        except Exception: id2clu = {}

    # flatten unique query strings (keep order per id)
    queries = []
    for id_, lst in id2all.items():
        for q in lst:
            if q and q not in queries:
                queries.append(q)
    for id_, lst in id2clu.items():
        for q in lst:
            if q and q not in queries:
                queries.append(q)

    if max_queries is not None:
        queries = queries[:max_queries]

    key = os.getenv("google_api_key") or os.getenv("GOOGLE_API_KEY")
    cx  = os.getenv("google_text_cse_id") or os.getenv("GOOGLE_CSE_ID")
    if not key or not cx:
        raise SystemExit("Missing google_api_key / google_text_cse_id environment vars.")

    svc = build("customsearch", "v1", developerKey=key)

    query2url = {}
    url2info  = {}

    for q in queries:
        try:
            resp = svc.cse().list(q=q, cx=cx, num=per_query).execute()
            items = resp.get("items") or []
        except Exception as e:
            print("CSE error on:", q, "->", repr(e))
            items = []

        urls = []
        for rank, it in enumerate(items, 1):
            link   = it.get("link")
            title  = it.get("title") or ""
            snip   = it.get("snippet") or ""
            if not link: continue
            urls.append(link)
            if link not in url2info:
                url2info[link] = {
                    "title": title,
                    "snippet": snip,
                    "provider": "google_text_search",
                    "query": q,
                    "rank": rank
                }
        query2url[q] = urls
        time.sleep(sleep)

    os.makedirs(base, exist_ok=True)
    save_json(query2url, f"{base}/query2url.json")
    save_json(url2info,  f"{base}/url2info.json")
    print("Wrote:", f"{base}/query2url.json", "and", f"{base}/url2info.json")
    print("queries:", len(queries), "urls:", len(url2info))
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UDK-VQA")
    ap.add_argument("--max_queries", type=int, default=30)
    ap.add_argument("--per_query",  type=int, default=3)
    ap.add_argument("--sleep",      type=float, default=0.25)
    args = ap.parse_args()
    main(**vars(args))

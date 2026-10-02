"""Flask server for the Scrabble AI move generator."""


import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import sys
import time

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from flask import Flask, jsonify, render_template, request

import json
import torch

from gaddag import GADDAG
from solver import Board, PREMIUM_MAP, LETTER_VALUES, generate_moves

from ai.model.policy_net import HybridPolicyNet
from ai.model.inference import PolicyInference
from ai.tile_tracker import TileTracker

app = Flask(__name__)

_gaddag_path = os.path.join(ROOT, "data", "gaddag.pkl")
if os.path.exists(_gaddag_path):
    print("Loading GADDAG …", end="", flush=True)
    _gaddag = GADDAG.load(_gaddag_path)
else:
    print("GADDAG not found, building from data/wordlist.txt (first run only) …", end="", flush=True)
    _gaddag = GADDAG()
    _gaddag.build_from_file(os.path.join(ROOT, "data", "wordlist.txt"))
    _gaddag.save(_gaddag_path)
print(f" {_gaddag.word_count:,} words.")

_inference = None
_model_path = os.path.join(ROOT, 'data', 'model.pt')
if os.path.exists(_model_path):
    try:
        model = HybridPolicyNet()
        ckpt  = torch.load(_model_path, map_location='cpu')
        model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt, strict=False)
        model.eval()
        _inference = PolicyInference(model)
        print("Model loaded.")
    except Exception as e:
        print(f"Model load failed: {e}")



_premium_js = json.dumps({r * 15 + c: p.name for (r, c), p in PREMIUM_MAP.items()})
_letter_values_js = json.dumps({k: v for k, v in LETTER_VALUES.items() if k != '?'})


@app.route("/")
def index():
    return render_template("index.html", premium_map=_premium_js, letter_values=_letter_values_js)


@app.route("/generate", methods=["POST"])
def generate():
    data  = request.get_json(force=True)
    tiles = data.get("board", [])
    rack  = data.get("rack", [])

    board = Board()
    for t in tiles:
        board.place_tile(t["row"], t["col"], t["letter"], t.get("is_blank", False))

    def new_tile_count(word, pos, pdir):
        n, ln = 0, len(word)
        for i in range(ln):
            r = pos[0]               if pdir == "H" else pos[0] - ln + 1 + i
            c = pos[1] - ln + 1 + i if pdir == "H" else pos[1]
            if 0 <= r < 15 and 0 <= c < 15 and board.grid[r][c] is None:
                n += 1
        return n

    rack_size = len(rack)
    our_score = data.get("our_score", 0)
    opp_score = data.get("opp_score", 0)
    turn      = data.get("turn", 0)

    out = []
    t = time.perf_counter()
    for word, score, pos, pdir in generate_moves(board, _gaddag.root, rack):
        out.append({
            "word":      word,
            "score":     score,
            "pos":       list(pos),
            "direction": pdir,
            "bingo":     rack_size == 7 and new_tile_count(word, pos, pdir) == 7,
            "equity":    None,
        })

    if _inference is not None and out:
        tracker = TileTracker()
        for r in range(15):
            for c in range(15):
                tile = board.grid[r][c]
                if tile is not None:
                    tracker.remove('?' if tile.is_blank else tile.letter)

        moves_tuples = [(m["word"], m["score"], tuple(m["pos"]), m["direction"]) for m in out]
        ranked = _inference.rank_moves(
            moves_tuples, board, rack, tracker,
            our_score, opp_score, turn, top_n=30,
            gaddag_root=_gaddag.root,
            lookahead_samples=5,
            lookahead_time=1.0,
        )
        equity_map = {(r[0][0], r[0][1], tuple(r[0][2]), r[0][3]): r[1] for r in ranked}
        for m in out:
            key = (m["word"], m["score"], tuple(m["pos"]), m["direction"])
            m["equity"] = round(equity_map[key] * 100, 1) if key in equity_map else None

        out.sort(key=lambda m: (m["equity"] is None, -(m["equity"] or 0)))

    print(f"Generated {len(out)} moves in {time.perf_counter() - t:.2f} seconds.")
    return jsonify(out[:100000])


if __name__ == "__main__":
    app.run(debug=False, port=8080)


# To Run: 
# http://127.0.0.1:8080
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.10", "matplotlib"]
# ///
"""Standalone scanline attn: global prefix + row strip -> valid tile closure.

No model weights, repository imports, training, or timing benchmark.
48 query / 12 KV heads; query heads 0,4,...,44 are global (NOT full KV groups).
The other heads use an 11-row full-width strip, then fill every active 128x128
metadata tile with VALID pairs. Internal invalid prefix slots and trailing
padding stay excluded, so tiles containing them can remain partial.

uv run flex_scanline.py --prefix 10 --height 32 --width 32
uv run flex_scanline.py --prefix 512 --valid-prefix 80
uv run flex_scanline.py --device cuda:0 --check

Outputs one PNG including prefix keys/queries, full token masks and tile maps.
CPU construction/visualization is quadratic and capped at 8192 padded tokens.
CUDA compiles mask construction. Cache build_scanline()'s returned BlockMask;
use torch.compile(flex_attention)(q,k,v,block_mask=bm,enable_gqa=True).
Coordinates are PATCHED attention-token rows, not raw VAE latent rows.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

BLOCK = 128
HEADS = 48
FULL_IDS = tuple(range(0, HEADS, 4))


def layout(height, width, prefix, valid_prefix=None, device="cpu"):
    """Row-major image after prefix slots, rounded to 256-token alignment.

    The prefix region is padded up to a 128-token tile boundary so prefix and
    image tokens never share a tile. Tile closure is tile-granular: a mixed
    tile would expand over image keys far outside the strip. Padding confines
    that over-expansion to dead prefix slots, which stay masked anyway.
    """
    valid_prefix = prefix if valid_prefix is None else valid_prefix
    if height < 1 or width < 1 or prefix < 0 or not 0 <= valid_prefix <= prefix:
        raise ValueError("Positive image dimensions and 0 <= valid_prefix <= prefix required")
    if prefix:
        prefix = (prefix + BLOCK - 1) // BLOCK * BLOCK
    end = prefix + height * width
    length = (end + 255) // 256 * 256
    idx = torch.arange(length, device=device)
    valid = ((idx < valid_prefix) | ((idx >= prefix) & (idx < end)))[None]
    rows = torch.zeros(1, length, dtype=torch.int32, device=device)
    rows[:, prefix:end] = torch.arange(height * width, device=device) // width
    return valid, rows, prefix


@torch.no_grad()
def build_scanline(valid, rows, prefix, window=11, *, return_source=False):
    """Production policy semantics, with batch-specific validity and owned captures.

    Build two unique heads (global/local), expand active tiles, reclassify, then
    map to 48 query heads. Unlike filling sparse lists blindly, reclassification
    retains invalid prefix/tag/image holes. No complete-KV-group restriction.
    Returns BlockMask, or (expanded, exact_strip) when return_source=True.
    """
    if (valid.ndim != 2 or rows.shape != valid.shape or valid.dtype != torch.bool
            or valid.device != rows.device or valid.shape[1] % BLOCK
            or window < 1 or window % 2 != 1 or not 0 <= prefix <= valid.shape[1]):
        raise ValueError("Need matching BxS rows/bool validity, aligned S, valid prefix, odd window")
    valid, rows = valid.detach().clone(), rows.detach().clone()
    batch, length = valid.shape
    radius = window // 2
    build = (torch.compile(create_block_mask, fullgraph=True, dynamic=False)
             if valid.is_cuda else create_block_mask)

    def strip(b, h, q, kv):
        return valid[b, q] & valid[b, kv] & (
            (h == 0) | (q < prefix) | (kv < prefix) |
            ((rows[b, q] - rows[b, kv]).abs() <= radius))

    source = build(strip, batch, 2, length, length, device=str(valid.device), BLOCK_SIZE=BLOCK)
    active = source.to_dense().bool().detach().clone()

    def expanded(b, h, q, kv):
        return valid[b, q] & valid[b, kv] & active[b, h, q // BLOCK, kv // BLOCK]

    tiled = build(expanded, batch, 2, length, length, device=str(valid.device), BLOCK_SIZE=BLOCK)
    selector = torch.ones(HEADS, dtype=torch.long, device=valid.device)
    selector[list(FULL_IDS)] = 0

    def map_heads(two, predicate):
        def mask_mod(b, h, q, kv):
            return predicate(b, selector[h], q, kv)
        return BlockMask.from_kv_blocks(
            two.kv_num_blocks.index_select(1, selector), two.kv_indices.index_select(1, selector),
            two.full_kv_num_blocks.index_select(1, selector), two.full_kv_indices.index_select(1, selector),
            BLOCK_SIZE=BLOCK, mask_mod=mask_mod, seq_lengths=(length, length))

    result = map_heads(tiled, expanded)
    return (result, map_heads(source, strip)) if return_source else result


def dense_head(bm, head=1):
    length = bm.seq_lengths[0]
    idx = torch.arange(length, device=bm.kv_indices.device)
    return bm.mask_mod(0, head, idx[:, None], idx[None, :]).cpu()


def tile_map(bm, head=1):
    active = bm.to_dense()[0, head].cpu().to(torch.uint8)
    counts, indices = bm.full_kv_num_blocks[0, head].cpu(), bm.full_kv_indices[0, head].cpu()
    for q, count in enumerate(counts):
        active[q, indices[q, :int(count)]] = 2
    return active


def check(device):
    """Small independent token reference; CUDA adds compiled GQA gradient parity."""
    for prefix, height, width, valid_prefix in ((0, 16, 64, 0), (10, 17, 16, 7)):
        valid, rows, prefix = layout(height, width, prefix, valid_prefix, device)
        valid = valid.expand(2, -1).clone()
        rows = rows.expand(2, -1).clone()
        valid[1, prefix + 9] = False  # arbitrary interior hole, not only suffix padding
        bm, source = build_scanline(valid, rows, prefix, return_source=True)
        length = valid.shape[1]
        idx = torch.arange(length, device=device)
        exact = (valid[:, :, None] & valid[:, None, :] &
                 ((idx[:, None] < prefix) | (idx[None, :] < prefix) |
                  ((rows[:, :, None] - rows[:, None, :]).abs() <= 5)))
        active = exact.reshape(2, length//BLOCK, BLOCK, length//BLOCK, BLOCK).any(4).any(2)
        expected = active.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
        expected &= valid[:, :, None] & valid[:, None, :]
        assert torch.equal(bm.to_dense(), source.to_dense())
        full = torch.zeros(HEADS, device=device, dtype=torch.bool)
        full[list(FULL_IDS)] = True
        reference = torch.where(full[None, :, None, None],
                                (valid[:, :, None] & valid[:, None, :])[:, None], expected[:, None])
        for b in range(2):
            for h in range(HEADS):
                actual = bm.mask_mod(b, h, idx[:, None], idx[None, :])
                assert torch.equal(actual, reference[b, h])
        if prefix == 0:
            assert bm.kv_num_blocks[0].sum() == 0
        valid.zero_(); rows.fill_(999)  # immutable captured buffers
        assert torch.equal(bm.mask_mod(0, 1, idx[:, None], idx[None, :]), expected[0])
        if str(device).startswith("cuda"):
            torch.manual_seed(123)
            inputs = [torch.randn(2, h, length, 32, device=device, requires_grad=True) for h in (48, 12, 12)]
            refs = [x.detach().clone().requires_grad_() for x in inputs]
            run = torch.compile(flex_attention, fullgraph=True)
            got = run(*inputs, block_mask=bm, enable_gqa=True, kernel_options={"BLOCK_M":32,"BLOCK_N":32})
            target = F.scaled_dot_product_attention(refs[0], refs[1].repeat_interleave(4,1),
                                                    refs[2].repeat_interleave(4,1), attn_mask=reference)
            torch.testing.assert_close(got, target, atol=2e-4, rtol=2e-4)
            grad = torch.randn_like(got)
            for a,b in zip(torch.autograd.grad(got,inputs,grad),torch.autograd.grad(target,refs,grad)):
                torch.testing.assert_close(a,b,atol=5e-4,rtol=5e-4)
        print(f"PASS prefix={prefix}, valid-prefix={valid_prefix}: spread heads, holes, immutable mask"
              + (", compiled GQA forward/backward" if str(device).startswith("cuda") else ""))


def visualize(bm, source, valid, height, width, prefix, window, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    valid = valid[0].cpu()
    length, end = len(valid), prefix + height*width
    exact, expanded = dense_head(source), dense_head(bm)
    # One prefix query and two image queries: first and center. Invalid prefix
    # queries are deliberately shown as empty, never silently made valid.
    queries = ([0] if prefix else []) + [prefix, prefix+(height//2)*width+width//2]
    fig = plt.figure(figsize=(16, 4*len(queries)+5), layout="constrained")
    outer = fig.add_gridspec(len(queries)+1, 1, height_ratios=[1]*len(queries)+[1.25])
    colors = ListedColormap(["#E9EDF0", "#239E98", "#444444"])
    for r,q in enumerate(queries):
        sub = outer[r].subgridspec(2, 2, height_ratios=[1,5])
        for c,(mask,label) in enumerate(((exact,"Exact full-width strip"),(expanded,"Tile-aligned scanline"))):
            ax = fig.add_subplot(sub[0,c])
            if prefix:
                slots = min(64, prefix)
                values = torch.full(((prefix+slots-1)//slots*slots,),2,dtype=torch.int64)
                values[:prefix] = torch.where(valid[:prefix],mask[q,:prefix].long(),2)
                ax.imshow(values.reshape(-1,slots),cmap=colors,vmin=0,vmax=2,aspect="auto",interpolation="nearest")
                if q < prefix: ax.plot(q%slots,q//slots,"s",color="#DC3545",markersize=5)
                ax.set(xlabel="Prefix key column (slots wrap at 64)",ylabel="Slot row")
            else:
                ax.text(.5,.5,"No prefix slots",ha="center",va="center");ax.set_axis_off()
            kind = "prefix" if q < prefix else f"image ({(q-prefix)//width},{(q-prefix)%width})"
            ax.set_title(f"{label} | query {q}: {kind} | valid={bool(valid[q])} | prefix keys {int(mask[q,:prefix].sum())}/{prefix}")
            ax = fig.add_subplot(sub[1,c])
            values = torch.where(valid[prefix:end],mask[q,prefix:end].long(),2).reshape(height,width)
            ax.imshow(values,cmap=colors,vmin=0,vmax=2,interpolation="nearest")
            if q >= prefix: ax.plot((q-prefix)%width,(q-prefix)//width,"s",color="#DC3545",markersize=5)
            if c==1:
                lo,hi=max(prefix,q//BLOCK*BLOCK),min(end,(q//BLOCK+1)*BLOCK)
                for yy in range(height):
                    a,b=max(lo-prefix,yy*width)-yy*width,min(hi-prefix,(yy+1)*width)-yy*width
                    if b>a: ax.add_patch(Rectangle((a-.5,yy-.5),b-a,1,fill=False,edgecolor="#DC3545",linewidth=.8))
            ax.set(title=f"Image keys: {int(mask[q,prefix:end].sum())}/{height*width}",xlabel="Image key column",ylabel="Image key row")
    sub = outer[-1].subgridspec(1,4)
    for c,(matrix,label,tiles) in enumerate(((exact,"Exact strip: token pairs",False),
                                            (expanded,"Expanded: token pairs",False),
                                            (tile_map(source),"Exact strip: tiles",True),
                                            (tile_map(bm),"Expanded: tiles",True))):
        ax=fig.add_subplot(sub[0,c]); n=len(matrix)
        if not tiles and n>512:
            factor=(n+511)//512
            matrix=F.avg_pool2d(matrix[None,None].float(),factor,ceil_mode=True,count_include_pad=False)[0,0]
        ax.imshow(matrix,cmap=ListedColormap(["#F0F0F0","#FF9F1C","#239E98"]) if tiles else "Greys",
                  vmin=0,vmax=2 if tiles else 1,interpolation="nearest",extent=(-.5,n-.5,n-.5,-.5))
        for boundary,color in ((prefix,"#DC3545"),(end,"#2464AD")):
            if 0<boundary<length:
                value=boundary/BLOCK-.5 if tiles else boundary-.5
                ax.axvline(value,color=color,linewidth=.9);ax.axhline(value,color=color,linewidth=.9)
        if tiles: label+=f"\nfull={int((matrix==2).sum())}, partial={int((matrix==1).sum())}"
        ax.set(title=label,xlabel="KV block" if tiles else "KV token",ylabel="Query block" if tiles else "Query token")
    fig.suptitle(f"Scanline policy: {window}-row strip → valid 128×128 tile closure | {height}×{width} image tokens\n"
                 f"Prefix={prefix} tile-aligned slots ({int(valid[:prefix].sum())} valid), trailing padding={length-end}; global query heads 0,4,…,44 / 48\n"
                 "Local head shown. Key views: teal=allowed, gray=blocked, dark=invalid; red=query/block outline.\n"
                 "Token plots: black=allowed density; red=prefix boundary, blue=padding boundary. Tile plots: orange=partial, teal=full.\n"
                 "Source: actual FlexAttention masks. Prefix participates before tile closure; invalid tokens always stay masked. No timing claims.",fontsize=12)
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=140);plt.close(fig)
    print(f"Saved {path}")


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--height",type=int,default=64);ap.add_argument("--width",type=int,default=64)
    ap.add_argument("--prefix",type=int,default=512)
    ap.add_argument("--valid-prefix",type=int,help="Only the first N prefix slots valid; default all")
    ap.add_argument("--window",type=int,default=11)
    ap.add_argument("--check",action="store_true",help="Correctness only; CUDA also checks gradients")
    ap.add_argument("--out",type=Path,default=Path("prefix-scanline.png"))
    args=ap.parse_args();torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit=64;torch._inductor.config.compile_threads=1
    if args.check: check(args.device);return
    valid,rows,prefix=layout(args.height,args.width,args.prefix,args.valid_prefix,args.device)
    if valid.shape[1]>8192: ap.error("Visualization is capped at8192 padded tokens")
    bm,source=build_scanline(valid,rows,prefix,args.window,return_source=True)
    visualize(bm,source,valid,args.height,args.width,prefix,args.window,args.out)


if __name__=="__main__":
    main()

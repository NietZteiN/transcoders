"""One-off (2026-09-08): does the text-only checkpoint reproduce the multimodal model's hidden states?"""
import sys, time, torch
sys.path.insert(0, "nla/src")
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma_text import ensure_text_checkpoint, load_gemma_text
m = "google/gemma-3-4b-it"
out = "data/nla/ml/gemma4b/host_text"
t = time.time(); ensure_text_checkpoint(m, out); print("materialise", round(time.time() - t), "s", flush=True)
tok = AutoTokenizer.from_pretrained(out)
t = time.time(); mm = load_gemma_text(out); print("text-only load", round(time.time() - t), "s", type(mm).__name__, round(sum(p.numel() for p in mm.parameters()) / 1e9, 3), "B params", flush=True)
print("embed_scale", mm.model.embed_tokens.embed_scale, "tied", mm.lm_head.weight.data_ptr() == mm.model.embed_tokens.weight.data_ptr())
x = tok("The quick brown fox jumps over the", return_tensors="pt")
with torch.no_grad(): out1 = mm(**x, output_hidden_states=True)
print("next tok:", repr(tok.decode(out1.logits[0, -1].argmax())))
print("mean-tok norms per hs idx", [round(float(h[0].float().norm(dim=-1).mean())) for h in out1.hidden_states])
cap = {}
h = mm.model.layers[33].register_forward_hook(lambda mod, i, o: cap.__setitem__('o', (o[0] if isinstance(o, tuple) else o)))
with torch.no_grad(): mm(**x)
h.remove()
print("hook L33 out == hs[34]?", torch.allclose(cap['o'].float(), out1.hidden_states[34].float()), " hs[34]==norm(hs[33])?", torch.allclose(out1.hidden_states[34].float(), mm.model.norm(out1.hidden_states[33]).float(), atol=0.5))
ids = tok.apply_chat_template([{"role": "user", "content": "What is 2+2? Answer with one number."}], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
with torch.no_grad(): g = mm.generate(**ids, max_new_tokens=8, do_sample=False)
print("chat reply:", repr(tok.decode(g[0, ids['input_ids'].shape[1]:])))
am = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16); print("auto ->", type(am).__name__)
with torch.no_grad(): out2 = am(**x, output_hidden_states=True)
print("hs max|diff| vs multimodal path:", [round(float((a.float() - b.float()).abs().max()), 3) for a, b in zip(out1.hidden_states, out2.hidden_states)])
del am
t = time.time(); cr = load_gemma_text(out, n_layers=6); print("truncated (6 layers) load", round(time.time() - t), "s", len(cr.model.layers))
cr.lm_head = torch.nn.Identity(); cr.model.norm = torch.nn.Identity()
with torch.no_grad(): o2 = cr(**x)
print("truncated output == full hs[6]?", torch.allclose(o2.logits.float(), out1.hidden_states[6].float()), o2.logits.shape)
# reload of the text-only checkpoint through AutoModelForCausalLM (what local_av.py does)
a2 = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16); print("AutoModelForCausalLM(text ckpt) ->", type(a2).__name__, "next tok:", repr(tok.decode(a2(**x).logits[0, -1].argmax())))

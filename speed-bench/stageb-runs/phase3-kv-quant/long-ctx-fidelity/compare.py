import json, glob, os, math, sys
SP=os.path.dirname(os.path.abspath(__file__))
def load(p):
    d=json.load(open(p))
    return d
def cosine(a,b):
    import array
    s=sa=sb=0.0
    for x,y in zip(a,b):
        s+=x*y; sa+=x*x; sb+=y*y
    return s/((sa**0.5)*(sb**0.5)+1e-30)
def softmax(v):
    m=max(v); ex=[math.exp(x-m) for x in v]; Z=sum(ex); return [e/Z for e in ex]
def topk(v,k):
    return [i for i,_ in sorted(enumerate(v),key=lambda t:t[1],reverse=True)[:k]]
def kl(p,q):  # KL(p||q)
    s=0.0
    for pi,qi in zip(p,q):
        if pi>0: s+=pi*math.log(pi/(qi+1e-30))
    return s
rows=[]
for q8f in sorted(glob.glob(os.path.join(SP,"q8","frontier_*.logits.json"))):
    base=os.path.basename(q8f)
    imf=os.path.join(SP,"imat",base)
    if not os.path.exists(imf): continue
    a=load(q8f); b=load(imf)
    la,lb=a["logits"],b["logits"]
    F=a["frontier_tokens"]
    cos=cosine(la,lb)
    pa,pb=softmax(la),softmax(lb)
    KL=kl(pa,pb)  # KL(Q8 || imat)
    t1=int(a["argmax_id"]==b["argmax_id"])
    ta,tb=set(topk(la,5)),set(topk(lb,5))
    t5=len(ta&tb)
    # max abs logit diff (aligned)
    mx=max(abs(x-y) for x,y in zip(la,lb))
    rows.append((F,cos,KL,t1,t5,a["argmax_id"],b["argmax_id"],mx,a.get("quality")))
print(f"{'ctx':>8} {'cosine':>9} {'KL(Q8||im)':>11} {'top1':>5} {'top5':>5} {'q8_arg':>7} {'im_arg':>7} {'maxabs':>8}")
for r in rows:
    F,cos,KL,t1,t5,qa,ia,mx,ql=r
    print(f"{F:>8} {cos:>9.5f} {KL:>11.5f} {('Y' if t1 else 'N'):>5} {t5:>4}/5 {qa:>7} {ia:>7} {mx:>8.3f}")
print(f"\n(quality/tensor-route flag in dumps: {rows[0][8] if rows else '?'} = prod default)")

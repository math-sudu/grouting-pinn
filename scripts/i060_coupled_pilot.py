"""Forward PINN feasibility pilot for pressure-controlled filtration.

Two complete PINNs share the same dimensionless model and initial/boundary
conditions. The resistance version eliminates Darcy pressure through its
spatial integral; this is an in-network physical constraint, not a reference
solver. Closure coefficients are specified examples, not fitted grout data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from torch.nn.functional import softplus

PHI0, CIN, SMAX, DISP, RATE, END = 0.4, 0.1, 0.2, 0.01, 4.0, 1.0


def mlp(inputs, outputs, width):
    layers = [torch.nn.Linear(inputs, width), torch.nn.Tanh()]
    for _ in range(2):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    layers += [torch.nn.Linear(width, outputs)]
    net = torch.nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    return net


class Fields(torch.nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.state = mlp(2, 3, width)
        self.pressure = mlp(2, 1, width)
        self.flow = mlp(1, 1, width)

    def forward(self, xt):
        x, t = xt[:, :1], xt[:, 1:]
        # y_x(1)=0 makes the physical concentration outlet gradient zero.
        y = x * (2 - x)
        raw = self.state(torch.cat((2*y-1, 2*t/END-1), 1))
        g = 1 - torch.exp(-(t/0.06)**2)
        c = CIN*g*torch.exp(-y*softplus(raw[:, :1]))
        sigma = SMAX*(1-torch.exp(-t*softplus(raw[:, 1:2])))
        diff_flux = CIN*DISP**0.5*(1-x)*g*raw[:, 2:3]
        return c, sigma, diff_flux

    def p(self, xt):
        # Used by differential/augmented runs. Resistance runs eliminate p.
        x, t = xt[:, :1], xt[:, 1:]
        return 1-x + t*x*(1-x)*self.pressure(2*xt-1)

    def q(self, times):
        return torch.exp(-times*softplus(self.flow(2*times/END-1)))


def grad(value, xt):
    return torch.autograd.grad(value.sum(), xt, create_graph=True)[0]


def permeability(sigma):
    return ((PHI0-sigma)/PHI0)**3


def resistance_q(net, times, order):
    z, w = leggauss(order)
    x = torch.as_tensor((z+1)/2)
    w = torch.as_tensor(w/2)
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    _, sigma, _ = net(torch.stack((xx.flatten(), tt.flatten()), 1))
    resistance = (w[:, None]/permeability(sigma).reshape(order, -1)).sum(0)
    return 1/resistance[:, None]


def residuals(net, x, times, method, order):
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    xt = torch.stack((xx.flatten(), tt.flatten()), 1).requires_grad_(True)
    c, sigma, diff_flux = net(xt)
    phi = PHI0-sigma
    qt = resistance_q(net, times, order) if method == "resistance" else net.q(times)
    q = qt.T.expand(len(x), -1).reshape(-1, 1)
    # q depends on time only; its spatial derivative is exactly zero.
    dc, ds = grad(c, xt), grad(sigma, xt)
    # The local derivative treats q(t) as spatially constant, as required.
    jx = q*dc[:, :1] + grad(diff_flux, xt)[:, :1]
    retention = RATE*q*c*(1-sigma/SMAX)
    mass = (phi*dc[:, 1:2]-c*ds[:, 1:2]+jx+retention)/CIN
    exchange = (ds[:, 1:2]-retention)/(RATE*CIN)
    constitutive = (diff_flux+phi*DISP*dc[:, :1])/(CIN*DISP**0.5)
    rows = dict(mass=mass, exchange=exchange, constitutive=constitutive)
    if method in ("differential", "augmented"):
        rows["darcy"] = q+permeability(sigma)*grad(net.p(xt), xt)[:, :1]
    if method == "augmented":
        # The integrated form constrains the same Darcy law. It exposes
        # pressure jumps that a finite set of local residuals can miss.
        rows["hydraulic_integral"] = qt/resistance_q(net, times, order)-1
    return rows


def objective(net, x, times, method, order):
    return sum(value.square().mean() for value in residuals(net,x,times,method,order).values())


def evaluate(net, method, order):
    x = torch.linspace(0, 1, 301)
    times = torch.linspace(0, END, 201)[:, None]
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    with torch.no_grad():
        c, sigma, df = net(torch.stack((xx.flatten(), tt.flatten()), 1))
        c, sigma, df = (v.reshape(len(x), len(times)) for v in (c, sigma, df))
        qr = resistance_q(net, times, max(64, order))
        q = resistance_q(net, times, order) if method == "resistance" else net.q(times)
        j = c*q.T+df
        mobile = torch.trapezoid((PHI0-sigma)*c, x, dim=0)
        deposited = torch.trapezoid(sigma, x, dim=0)
        balance = mobile+deposited-torch.cat((torch.zeros(1),torch.cumsum(
            ((j[0,1:]-j[-1,1:])+(j[0,:-1]-j[-1,:-1]))*(END/200)/2,0)))
    # Independent midpoint grid across the physical space-time domain.
    xv = (torch.arange(96)+0.5)/96
    tv = ((torch.arange(80)+0.5)*END/80)[:, None]
    rv = residuals(net,xv,tv,method,64)
    metrics = {key+"_rms":float(v.detach().square().mean().sqrt()) for key,v in rv.items()}
    metrics.update(q_final=float(q[-1]), outlet_c_final=float(c[-1,-1]),
                   sigma_max=float(sigma.max()), concentration_min=float(c.min()),
                   concentration_max=float(c.max()), deposited_final=float(deposited[-1]),
                   mobile_final=float(mobile[-1]), max_mass_defect=float(balance.abs().max()),
                   max_pressure_flow_defect=float((q-qr).abs().max()),
                   cumulative_inlet_solid=float(torch.trapezoid(j[0],times.flatten())),
                   cumulative_outlet_solid=float(torch.trapezoid(j[-1],times.flatten())))
    fields = {"x":x.numpy(),"times":times.flatten().numpy(),"c":c.numpy(),
              "sigma":sigma.numpy(),"q":q.numpy(),"j":j.numpy(),"mass_defect":balance.numpy()}
    return metrics, fields


def check():
    torch.manual_seed(7)
    net = Fields()
    t = torch.linspace(0,END,11)[:,None]
    x = torch.linspace(0,1,11)[:,None]
    c0,s0,j0 = net(torch.cat((x,0*x),1))
    inlet,_,_ = net(torch.cat((0*t,t),1))
    out = torch.cat((0*t+1,t),1).requires_grad_(True)
    co,_,jo = net(out)
    assert torch.equal(c0,0*c0) and torch.equal(s0,0*s0)
    assert torch.allclose(inlet,CIN*(1-torch.exp(-(t/.06)**2)))
    assert grad(co,out)[:,0].abs().max() == 0 and jo.abs().max() == 0
    # A directional finite difference checks the hydraulic feedback gradient.
    q = resistance_q(net,t,48).sum()
    par = net.state[-1].bias
    dq = torch.autograd.grad(q,par)[0][1].item()
    with torch.no_grad():
        old=par[1].item(); eps=1e-5
        par[1]=old+eps; plus=resistance_q(net,t,48).sum().item()
        par[1]=old-eps; minus=resistance_q(net,t,48).sum().item()
        par[1]=old
    fd=(plus-minus)/(2*eps)
    assert abs(fd-dq)<1e-7
    q48=resistance_q(net,t,48); q96=resistance_q(net,t,96)
    print(json.dumps(dict(initial_inlet_outlet="passed",flow_gradient=dq,
                          flow_gradient_fd=fd,quadrature_difference=float((q48-q96).detach().abs().max()))))


def train(args):
    torch.manual_seed(args.seed)
    net=Fields(args.width)
    gen=torch.Generator().manual_seed(args.seed+1000)
    def samples():
        if args.sampling == "stratified":
            # Cover the whole physical interval; the old sparse random time
            # set admitted unobserved deposition jumps and pressure drift.
            x=torch.cat(((torch.arange(32)+torch.rand(32,generator=gen))/32,
                         torch.rand(8,generator=gen)*.15))
            times=torch.cat((((torch.arange(64)[:,None]+torch.rand(64,1,generator=gen))/64)*END,
                             torch.rand(16,1,generator=gen)*.15,
                             torch.tensor([[0.0],[END]])))
            return x,times
        x=torch.cat((torch.rand(24,generator=gen),torch.rand(8,generator=gen)*.15))
        times=torch.cat((torch.rand(16,1,generator=gen)*END,torch.rand(8,1,generator=gen)*.15))
        return x,times
    x,t=samples()
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    start=time.perf_counter(); history=[]
    for step in range(args.adam):
        if step%200==0: x,t=samples()
        optimizer.zero_grad(set_to_none=True)
        loss=objective(net,x,t,args.method,args.order)
        loss.backward(); optimizer.step()
        if step%1000==0 or step==args.adam-1:
            row=dict(step=step+1,loss=float(loss.detach()),seconds=time.perf_counter()-start)
            history.append(row); print(json.dumps(row),flush=True)
    opt=torch.optim.LBFGS(net.parameters(),max_iter=args.lbfgs,history_size=50,
                         line_search_fn="strong_wolfe",tolerance_grad=1e-10,tolerance_change=1e-12)
    evaluations=0
    def closure():
        nonlocal evaluations
        opt.zero_grad(set_to_none=True)
        loss=objective(net,x,t,args.method,args.order)
        loss.backward(); evaluations+=1
        return loss
    if args.lbfgs: opt.step(closure)
    elapsed=time.perf_counter()-start
    metrics,fields=evaluate(net,args.method,args.order)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'fields.npz',**fields)
    torch.save(net.state_dict(),args.output/'model.pt')
    config=vars(args).copy(); config['output']=str(args.output)
    result=dict(config=config,model=dict(phi0=PHI0,cin=CIN,smax=SMAX,D=DISP,rate=RATE,T=END,
                pressure_drop=1.0,viscosity=1.0,coefficient_role="specified dimensionless examples"),
                metrics=metrics,seconds=elapsed,lbfgs_evaluations=evaluations,history=history,
                runtime=dict(torch=torch.__version__,dtype="float64",device="cpu"))
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(method=args.method,seed=args.seed,metrics=metrics,seconds=elapsed)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['differential','resistance','augmented'],default='resistance')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--order',type=int,default=32)
    parser.add_argument('--adam',type=int,default=4000)
    parser.add_argument('--lbfgs',type=int,default=600)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--sampling',choices=['random','stratified'],default='stratified')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('results/i060_coupled/pilot'))
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    check() if args.check else train(args)

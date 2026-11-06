# -*- coding: utf-8 -*-
"""Reproducible gravity forward-modeling toolkit for Marajo Basin research.

Author-origin source material: Nelson Ribeiro-Filho (see original_codes/).
New research implementation prepared for method comparison. Not a peer-reviewed
or independently certified geophysical library.

Conventions
-----------
Cartesian x,y,z in metres; z POSITIVE DOWN; density contrast in kg/m**3;
returned g_z in mGal, POSITIVE DOWN (attraction). Coordinates must be projected
in the same metre-based CRS. Do not feed longitude/latitude as x/y.

Geometry: analytical rectangle (Nagy/Plouff corner sum), exact polygon contour
formula (valid for vertical sides and horizontal top/bottom), volume GL quadrature,
polygon area GL quadrature (for independent verification), spectral FFT for
constant-thickness horizontal slabs on a regular grid, spherical tesseroid GLQ,
column-based layered Moho/topography model, and research statistics.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import json
import math
import time
import warnings
import numpy as np

G = 6.67430e-11  # CODATA 2018 recommended (SI)
SI_TO_MGAL = 1.0e5


def _observations(x, y, z):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    z = np.broadcast_to(np.asarray(z, float), x.shape)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError('Observation coordinates must be finite')
    return x, y, z


def _density(rho, density_unit='kg/m3'):
    if density_unit in ('kg/m3', 'kg/m^3'):
        return float(rho)
    if density_unit in ('g/cm3', 'g/cm^3'):
        return float(rho)*1000.
    raise ValueError('density_unit must be kg/m3 or g/cm3')


@lru_cache(maxsize=40)
def _leggauss(order):
    if order < 1 or order > 128:
        raise ValueError('quadrature order must be between 1 and 128')
    return np.polynomial.legendre.leggauss(order)


def _bounds(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (6,) or not np.isfinite(b).all() or not np.all(b[[1, 3, 5]] > b[[0, 2, 4]]):
        raise ValueError('bounds must be [xmin,xmax,ymin,ymax,zt,zb] with increasing limits')
    return b


def _corner_kernel(x, y, z):
    """Primitive for exact rectangle gravity; z is relative to observation."""
    r = np.sqrt(x*x + y*y + z*z)
    # At boundaries these logs may approach zero; the multiplied limit is zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(x == 0., 0., x*np.log(np.maximum(r+y, np.finfo(float).tiny)))
        t2 = np.where(y == 0., 0., y*np.log(np.maximum(r+x, np.finfo(float).tiny)))
    # arctan(x*y/(z*r)), using arctan2 then principal branch [-pi/2,pi/2]
    a = np.arctan2(x*y, z*r)
    a = np.where(a > np.pi/2, a - np.pi, a)
    a = np.where(a < -np.pi/2, a + np.pi, a)
    return t1 + t2 - z*a


def gz_rectangular(x, y, z, bounds, rho, density_unit='kg/m3'):
    """Exact gravity of a homogeneous, axis-aligned rectangular prism [mGal]."""
    x,y,z = _observations(x,y,z)
    b=_bounds(bounds)
    xx=[b[0]-x,b[1]-x]; yy=[b[2]-y,b[3]-y]; zz=[b[4]-z,b[5]-z]
    out=np.zeros(x.shape,float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                out += ((-1.)**(i+j+k))*_corner_kernel(xx[i],yy[j],zz[k])
    return G*_density(rho,density_unit)*SI_TO_MGAL*out


def gz_rectangular_quadrature(x, y, z, bounds, rho, order=6, density_unit='kg/m3'):
    """Independent 3D Gauss-Legendre volume integral; observation must lie outside volume."""
    x,y,z=_observations(x,y,z); b=_bounds(bounds)
    if np.any((x>=b[0])&(x<=b[1])&(y>=b[2])&(y<=b[3])&(z>=b[4])&(z<=b[5])):
        raise ValueError('Quadrature verification excludes stations inside the source volume')
    v,w=_leggauss(int(order))
    vals=[(v*(b[2*i+1]-b[2*i])+(b[2*i+1]+b[2*i]))/2 for i in range(3)]
    ws=[w*(b[2*i+1]-b[2*i])/2 for i in range(3)]
    total=np.zeros_like(x,float)
    for i,a in enumerate(vals[0]):
        for j,c in enumerate(vals[1]):
            for k,d in enumerate(vals[2]):
                r2=(a-x)**2+(c-y)**2+(d-z)**2
                total += ws[0][i]*ws[1][j]*ws[2][k]*(d-z)/np.power(r2,1.5)
    return total*G*_density(rho,density_unit)*SI_TO_MGAL


def signed_polygon_area(vertices):
    v=np.asarray(vertices,float)
    if v.ndim != 2 or v.shape[1]!=2 or len(v)<3 or not np.isfinite(v).all():
        raise ValueError('vertices must be finite (N,2), N>=3')
    return .5*float(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1)))


def _ccw_vertices(vertices):
    v=np.asarray(vertices,float)
    # Disallow explicitly repeated last vertex, duplicative edge contributes zero.
    if len(v)>3 and np.allclose(v[0],v[-1],rtol=0,atol=1.e-12):
        v=v[:-1]
    a=signed_polygon_area(v)
    if abs(a)<1.e-20:
        raise ValueError('degenerate polygon')
    return v if a>0 else v[::-1].copy()


def regular_polygon(xc,yc,radius,n=6,rotation_deg=0.):
    if int(n)<3 or radius<=0: raise ValueError('n>=3 and radius>0')
    t=np.deg2rad(rotation_deg)+np.arange(n)*2*np.pi/n
    return np.column_stack((xc+radius*np.cos(t),yc+radius*np.sin(t)))


def _edge_primitive(s,d,h):
    """Primitive of sqrt(s²+d²+h²)/(s²+d²) - abs(h)/(s²+d²).
    Valid for nonzero signed d. Derived by integrating the 2D Green flux.
    """
    dd=abs(d); hh=abs(h)
    a=np.sqrt(d*d+hh*hh)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.arcsinh(s/a) + (hh/dd)*(
            np.arctan2(hh*s, dd*np.sqrt(s*s+a*a)) - np.arctan(s/dd)))


def _polygon_contour_integral(vertices, x, y, h):
    """Integral of 1/sqrt((x'-x)^2+(y'-y)^2+h^2) over polygon.

    Applies Green's theorem to an analytic edge primitive. CCW outward normals.
    h>0 required for regular evaluation (observation outside vertical prism).
    """
    vv=_ccw_vertices(vertices)
    x,y,h=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(h,float))
    # h=0 is allowed away from polygon edge singular configurations.
    result=np.zeros_like(x,float)
    for a,b in zip(vv,np.roll(vv,-1,axis=0)):
        e=b-a; length=float(np.hypot(*e))
        if length==0: continue
        tangent=e/length; outward=np.array([tangent[1],-tangent[0]])
        delta_x=a[0]-x; delta_y=a[1]-y
        d=delta_x*outward[0]+delta_y*outward[1]
        s0=delta_x*tangent[0]+delta_y*tangent[1]
        s1=s0+length
        # d~0 => contour flux of this particular side tends to 0 when h!=0.
        safe=np.abs(d)>1.e-9
        if np.any(safe):
            ds=np.where(safe,d,1.)
            term=ds*(_edge_primitive(s1,ds,h)-_edge_primitive(s0,ds,h))
            result+=np.where(safe,term,0.)
    return result


def gz_polygon(x,y,z,vertices,top,bottom,rho,density_unit='kg/m3'):
    """Analytical edge-sum vertical gravity of vertical polygonal prism [mGal].

    Polygon can be convex or concave but must be simple (no self intersections).
    Horizontal top/bottom. Boundary-coincident stations need special validation.
    The vertical primitive is integrated exactly and Green's theorem reduces
    the remaining area integral to an analytically integrated perimeter sum.
    """
    x,y,z=_observations(x,y,z)
    if not bottom>top: raise ValueError('bottom must exceed top, positive downward')
    # Valid also for interior z, using the signed vertical primitive.
    t=_polygon_contour_integral(vertices,x,y,top-z)
    b=_polygon_contour_integral(vertices,x,y,bottom-z)
    return G*_density(rho,density_unit)*SI_TO_MGAL*(t-b)


def gz_hexagonal(x,y,z,xc,yc,radius,top,bottom,rho,rotation_deg=0.,density_unit='kg/m3'):
    """Exact hexagonal prism gravity via analytical polygon contour [mGal]."""
    return gz_polygon(x,y,z,regular_polygon(xc,yc,radius,6,rotation_deg),top,bottom,rho,density_unit)


def _cross2(u,v):
    return u[...,0]*v[...,1] - u[...,1]*v[...,0]


def _point_in_triangle(p,a,b,c, tol=1.e-10):
    ab=_cross2(b-a,p-a);bc=_cross2(c-b,p-b);ca=_cross2(a-c,p-c)
    return (ab>=-tol and bc>=-tol and ca>=-tol)


def triangulate_polygon(vertices):
    """Ear clipping of simple CCW/clockwise polygon; returns (n-2,3,2)."""
    v=_ccw_vertices(vertices)
    ids=list(range(len(v))); triangles=[]; guard=0
    while len(ids)>3:
        guard+=1
        if guard>len(v)**2: raise ValueError('Polygon cannot be triangulated (self-intersection?)')
        found=False
        for i in range(len(ids)):
            i0,i1,i2=ids[i-1],ids[i],ids[(i+1)%len(ids)]
            a,b,c=v[[i0,i1,i2]]
            if _cross2(b-a,c-b)<=1.e-12:continue
            if any(_point_in_triangle(v[q],a,b,c) for q in ids if q not in (i0,i1,i2)):
                continue
            triangles.append([a,b,c]);ids.pop(i);found=True;break
        if not found:raise ValueError('Invalid polygon for ear clipping')
    triangles.append(v[ids])
    return np.asarray(triangles,float)


def gz_polygon_quadrature(x,y,z,vertices,top,bottom,rho,order=12,density_unit='kg/m3'):
    """Independent 2D GLQ on triangular partition, exact vertical integration."""
    x,y,z=_observations(x,y,z)
    if not bottom>top or np.any((z>=top)&(z<=bottom)):
        raise ValueError('stations outside top and bottom required')
    q,w=_leggauss(int(order)); q=(q+1)*.5; w=w*.5
    out=np.zeros_like(x,float)
    for a,b,c in triangulate_polygon(vertices):
        det=abs(_cross2(b-a,c-a))
        for u,wu in zip(q,w):
            for v,wv in zip(q,w):
                p=a+u*(b-a)+(1-u)*v*(c-a)
                r2=(p[0]-x)**2+(p[1]-y)**2
                h1=top-z;h2=bottom-z
                out+=det*(1-u)*wu*wv*(1/np.sqrt(r2+h1*h1)-1/np.sqrt(r2+h2*h2))
    return out*G*_density(rho,density_unit)*SI_TO_MGAL


def gz_tesseroid(radius,latitude,longitude,r1,r2,lat1,lat2,lon1,lon2,rho,order=4,density_unit='kg/m3'):
    """Radial inward gravity [mGal] of a spherical tesseroid via Gauss-Legendre.

    All lengths [m], latitude/longitude [radians]. 'g_z' is LOCAL RADIAL INWARD,
    not global Cartesian z; local planar approximation must be justified.
    Stations cannot lie in the volume. Increase order/subdivide for near-field.
    """
    if not(0<r1<r2 and lat1<lat2 and lon1<lon2):raise ValueError('Invalid tesseroid bounds')
    r,lat,lon=np.broadcast_arrays(np.asarray(radius,float),np.asarray(latitude,float),np.asarray(longitude,float))
    q,w=_leggauss(int(order)); limits=((r1,r2),(lat1,lat2),(lon1,lon2))
    nodes=[(q*(b-a)+(b+a))/2 for a,b in limits]
    weights=[w*(b-a)/2 for a,b in limits]
    out=np.zeros_like(r,float)
    for i,rs in enumerate(nodes[0]):
        for j,ps in enumerate(nodes[1]):
            for k,ls in enumerate(nodes[2]):
                one_minus_cos=2*(np.sin((lat-ps)/2)**2 +
                    np.cos(lat)*np.cos(ps)*np.sin((lon-ls)/2)**2)
                dist2=(r-rs)**2+2*r*rs*one_minus_cos
                if np.any(dist2<=0):raise ValueError('Observation intersects mass quadrature point')
                numerator=(r-rs)+rs*one_minus_cos
                out+=weights[0][i]*weights[1][j]*weights[2][k]*rs**2*np.cos(ps)*numerator/np.power(dist2,1.5)
    return G*_density(rho,density_unit)*SI_TO_MGAL*out


def gz_tesseroid_adaptive(radius,latitude,longitude,r1,r2,lat1,lat2,lon1,lon2,rho,
                           order=4,subdivision_ratio=1.5,max_subdivisions=48):
    """Adaptive-ish angular subdivisions based on nearest station separation.

    Uses a conservative angular cell-length / depth-distance control.
    Not a rigorous a posteriori error estimate; refine order and ratio and
    compare to a high-resolution independent reference for publication.
    """
    rad=np.asarray(radius,float)
    if np.any(rad<=r2):
        raise ValueError('Adaptive tesseroid presently requires observations above body')
    height=float(np.min(rad)-r2)
    horizontal=max(float(np.mean([r1,r2])),1.)
    maxwidth=height*subdivision_ratio
    nlat=min(max_subdivisions,max(1,int(np.ceil(horizontal*(lat2-lat1)/maxwidth))))
    nlon=min(max_subdivisions, max(1, int(np.ceil(horizontal*abs(np.cos((lat1+lat2)/2))*(lon2-lon1)/maxwidth))))
    out=np.zeros(np.broadcast_arrays(np.asarray(radius),np.asarray(latitude),np.asarray(longitude))[0].shape,float)
    lat_edges=np.linspace(lat1,lat2,nlat+1);lon_edges=np.linspace(lon1,lon2,nlon+1)
    for i in range(nlat):
        for j in range(nlon):
            out+=gz_tesseroid(radius,latitude,longitude,r1,r2,lat_edges[i],lat_edges[i+1],
                              lon_edges[j],lon_edges[j+1],rho,order=order)
    return out


def gz_fft_slab(rho_grid,dx,dy,top,bottom,observation_z=0.,pad_factor=4,density_unit='kg/m3'):
    """Approximate periodic spectral solution for a flat slab with horizontal rho variations.

    rho_grid[n_y,n_x], equal spacing [m], uniformly spaced cell centers.
    Zero-padding reduces periodic image contamination; finite-domain and sampling
    errors remain. Not valid for spatially varying top/bottom or scattered data.
    """
    rho=np.asarray(rho_grid,float)
    if rho.ndim!=2 or not np.isfinite(rho).all(): raise ValueError('rho_grid must be finite 2D')
    if min(dx,dy)<=0 or not observation_z<top<bottom or pad_factor<2:
        raise ValueError('positive dx/dy, observation_z<top<bottom, pad_factor>=2 required')
    ny,nx=rho.shape
    py,px=int(ny*pad_factor),int(nx*pad_factor)
    if py<ny or px<nx:raise ValueError('padding invalid')
    padded=np.zeros((py,px)); y0=(py-ny)//2;x0=(px-nx)//2
    padded[y0:y0+ny,x0:x0+nx]=rho*_density(1.,density_unit)
    ky=2*np.pi*np.fft.fftfreq(py,dy)[:,None]
    kx=2*np.pi*np.fft.fftfreq(px,dx)[None,:]
    k=np.sqrt(kx*kx+ky*ky)
    h1=top-observation_z;h2=bottom-observation_z
    # Stable expression (exp(-k*h1) - exp(-k*h2)) / k.
    transfer=np.exp(-k*h1)*(-np.expm1(-k*(h2-h1)))/np.where(k==0.,1.,k)
    transfer[0,0]=h2-h1
    # Footprint of each constant-density cell (finite horizontal sampling).
    transfer*=np.sinc(kx*dx/(2*np.pi))*np.sinc(ky*dy/(2*np.pi))
    gz=2*np.pi*G*np.fft.ifft2(np.fft.fft2(padded)*transfer).real*SI_TO_MGAL
    return gz[y0:y0+ny,x0:x0+nx]


@dataclass
class Cell:
    vertices: np.ndarray  # CCW polygon; for rectangles, bounds in helper method
    top: float
    bottom: float
    density: float  # density contrast [kg/m3]
    def __post_init__(self):
        self.vertices=_ccw_vertices(self.vertices)
        if not self.top<self.bottom:raise ValueError('invalid cell thickness')
    @property
    def volume(self): return abs(signed_polygon_area(self.vertices))*(self.bottom-self.top)
    def rectangle_bounds(self):
        v=self.vertices
        if len(v)!=4:raise ValueError('not four vertices')
        xmin,xmax=v[:,0].min(),v[:,0].max();ymin,ymax=v[:,1].min(),v[:,1].max()
        if not np.allclose(abs(signed_polygon_area(v)),(xmax-xmin)*(ymax-ymin)):
            raise ValueError('non-axis-aligned cell')
        return [xmin,xmax,ymin,ymax,self.top,self.bottom]


def make_rectangular_cells(xcenters,ycenters,top,base,dx,dy,densities=(300.,),fractions=None):
    """Column layers between sampled surfaces. x,y,top,base arrays SAME shape.

    Top/base are absolute positive-down coordinates, not elevations with different
    datum. 'fractions' monotone subdivision 0,...,1; density one per interval.
    """
    arrays=np.broadcast_arrays(np.asarray(xcenters,float),np.asarray(ycenters,float),np.asarray(top,float),np.asarray(base,float))
    xs,ys,ts,bs=[v.ravel() for v in arrays]
    if min(dx,dy)<=0:raise ValueError('dx/dy must be positive')
    dens=np.atleast_1d(densities).astype(float)
    if fractions is None:fractions=np.linspace(0,1,len(dens)+1)
    fr=np.asarray(fractions,float)
    if len(fr)!=len(dens)+1 or fr[0]!=0 or fr[-1]!=1 or np.any(np.diff(fr)<=0):
        raise ValueError('fractions must be 0..1 with n_layers+1 entries')
    cells=[]
    for x,y,t,b in zip(xs,ys,ts,bs):
        if not np.isfinite([x,y,t,b]).all():continue
        if b<=t:continue
        verts=np.array([[x-dx/2,y-dy/2],[x+dx/2,y-dy/2],
                        [x+dx/2,y+dy/2],[x-dx/2,y+dy/2]])
        for j,rho in enumerate(dens):
            cells.append(Cell(verts,t+(b-t)*fr[j],t+(b-t)*fr[j+1],float(rho)))
    return cells


def make_regular_hex_cells(xmin,xmax,ymin,ymax,radius,top,base,densities=(300.,),fractions=None,clip_boundary=False):
    """Hex grid with polygon clipping to rectangle (shapely optional if clipping).

    `top` and `base` are scalars only here; interpolated interfaces need a
    separately supplied observation-based geological model, not implicit guesses.
    """
    if radius<=0 or not xmin<xmax or not ymin<ymax or not top<base:raise ValueError('invalid parameters')
    if not clip_boundary:
        warnings.warn('Hex polygons extend beyond bbox; use clip_boundary=True for matched footprint')
    if clip_boundary:
        try:
            from shapely.geometry import Polygon,box
        except ImportError as exc:raise ImportError('pip install shapely for clipped hexagons') from exc
        boundary=box(xmin,ymin,xmax,ymax)
    dy=np.sqrt(3)*radius; dx=1.5*radius
    cols=int(np.ceil((xmax-xmin)/dx))+3;rows=int(np.ceil((ymax-ymin)/dy))+3
    dens=np.atleast_1d(densities).astype(float)
    fr=np.linspace(0,1,len(dens)+1) if fractions is None else np.asarray(fractions,float)
    if len(fr)!=len(dens)+1 or fr[0]!=0 or fr[-1]!=1 or np.any(np.diff(fr)<=0):raise ValueError('invalid fractions')
    cells=[]
    for i in range(-2,cols):
        cx=xmin+(i+.25)*dx
        for j in range(-2,rows):
            cy=ymin+(j+.25)*dy+(i%2)*dy/2
            vv=regular_polygon(cx,cy,radius,6)
            if clip_boundary:
                intersection=Polygon(vv).intersection(boundary)
                if intersection.is_empty:continue
                pieces=list(intersection.geoms) if intersection.geom_type=='MultiPolygon' else [intersection]
                verts_list=[np.asarray(p.exterior.coords[:-1]) for p in pieces if p.area>1e-8]
            else:verts_list=[vv] if xmin<=cx<=xmax and ymin<=cy<=ymax else []
            for vertices in verts_list:
                for q,rho in enumerate(dens):
                    cells.append(Cell(vertices,top+(base-top)*fr[q],top+(base-top)*fr[q+1],rho))
    return cells


def gz_cells(x,y,z,cells,method='polygon'):
    """Forward sum. Methods: polygon, rectangular, polygon_quadrature."""
    x,y,z=_observations(x,y,z)
    total=np.zeros_like(x,float)
    for c in cells:
        if method=='polygon':
            total+=gz_polygon(x,y,z,c.vertices,c.top,c.bottom,c.density)
        elif method=='rectangular':
            total+=gz_rectangular(x,y,z,c.rectangle_bounds(),c.density)
        elif method=='polygon_quadrature':
            total+=gz_polygon_quadrature(x,y,z,c.vertices,c.top,c.bottom,c.density)
        else:raise ValueError('unknown method')
    return total


def regional_residual(observed_bouguer,regional_model):
    """Observed Bouguer anomaly minus compatible regional response [mGal].

    Caller MUST verify same reference, normal gravity and terrain/Bouguer
    reductions. Do not double-correct topography.
    """
    obs=np.asarray(observed_bouguer,float); reg=np.asarray(regional_model,float)
    if obs.shape!=reg.shape:raise ValueError('data and regional must have same shape')
    return obs-reg


def compare_fields(a,b):
    """Difference statistics [mGal] and Pearson R. No fabricated uncertainty."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape:raise ValueError('same shapes required')
    ok=np.isfinite(a)&np.isfinite(b)
    if not ok.any():raise ValueError('no valid paired values')
    aa=a[ok];bb=b[ok];diff=aa-bb
    cc=float(np.corrcoef(aa,bb)[0,1]) if len(aa)>1 and np.std(aa)>0 and np.std(bb)>0 else float('nan')
    return {'n':int(len(aa)),'bias_mGal':float(np.mean(diff)),
            'rmse_mGal':float(np.sqrt(np.mean(diff**2))),
            'mae_mGal':float(np.mean(np.abs(diff))),
            'max_abs_mGal':float(np.max(np.abs(diff))),
            'pearson_r':cc}


def upward_continuation_fft(field,dx,dy,height,pad_factor=3):
    """Fourier upward continuation on a REGULAR grid; boundary padding approximate."""
    a=np.asarray(field,float)
    if a.ndim!=2 or not np.isfinite(a).all() or height<0 or min(dx,dy)<=0:raise ValueError('invalid grid/height')
    ny,nx=a.shape;py,px=ny*pad_factor,nx*pad_factor
    padded=np.pad(a,(((py-ny)//2,py-ny-(py-ny)//2),((px-nx)//2,px-nx-(px-nx)//2)),mode='edge')
    k=np.hypot((2*np.pi*np.fft.fftfreq(py,dy))[:,None],(2*np.pi*np.fft.fftfreq(px,dx))[None,:])
    out=np.fft.ifft2(np.fft.fft2(padded)*np.exp(-height*k)).real
    oy=(py-ny)//2;ox=(px-nx)//2
    return out[oy:oy+ny,ox:ox+nx]


def polynomial_regional(x,y,observed,degree=2):
    """Least squares 2D polynomial, centered/scaled; returns regional and residual."""
    if degree<0 or degree>6:raise ValueError('degree must be 0..6')
    x,y,v=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(observed,float))
    mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(v)
    if mask.sum()<((degree+1)*(degree+2))//2:raise ValueError('insufficient finite observations')
    xx=(x-np.mean(x[mask]))/(np.std(x[mask]) or 1)
    yy=(y-np.mean(y[mask]))/(np.std(y[mask]) or 1)
    cols=[(xx**i)*(yy**j) for i in range(degree+1) for j in range(degree+1-i)]
    A=np.stack([a[mask] for a in cols],axis=1)
    coef=np.linalg.lstsq(A,v[mask],rcond=None)[0]
    reg=np.full_like(v,np.nan);reg[mask]=A@coef
    return reg,v-reg,coef


def load_xyz_csv(path,x='x_m',y='y_m',z='z_m',value='bouguer_mGal',delimiter=','):
    """Read sparse observation XYZ CSV; no automatic interpolation or CRS guessing."""
    a=np.genfromtxt(str(path),names=True,delimiter=delimiter,dtype=float,encoding=None)
    if a.size==0:raise ValueError('empty CSV')
    return tuple(np.atleast_1d(a[key]).astype(float) for key in [x,y,z,value])


def save_xyz_csv(path,x,y,z,fields):
    x,y,z=_observations(x,y,z)
    data=[x.ravel(),y.ravel(),z.ravel()]
    names=['x_m','y_m','z_m']
    for name,field in fields.items():
        f=np.asarray(field,float)
        if f.shape!=x.shape:raise ValueError(f'{name} shape != coordinates')
        names.append(name);data.append(f.ravel())
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(str(path),np.column_stack(data),delimiter=',',header=','.join(names),comments='',fmt='%.10g')


def synthetic_experiment(output_dir,make_figures=True):
    """Tiny reproducible SANITY experiment, NOT real Marajo gravity results."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(-2000,2000,17); xx,yy=np.meshgrid(axis,axis,indexing='xy')
    zz=np.zeros_like(xx)
    rect=[-750,750,-650,650,900,2700]
    vertices=np.array([[rect[0],rect[2]],[rect[1],rect[2]],
                       [rect[1],rect[3]],[rect[0],rect[3]]])
    ref=gz_rectangular(xx,yy,zz,rect,350.)
    contour=gz_polygon(xx,yy,zz,vertices,900,2700,350.)
    glq=gz_rectangular_quadrature(xx,yy,zz,rect,350.,order=8)
    metrics={'polygon_vs_exact':compare_fields(contour,ref),
             'glq8_vs_exact':compare_fields(glq,ref)}
    save_xyz_csv(out/'synthetic_gz.csv',xx,yy,zz,
                 {'exact_rectangle_mGal':ref,'polygon_mGal':contour,
                  'GLQ8_mGal':glq,'polygon_minus_rectangle_mGal':contour-ref})
    if make_figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i,(data,label) in enumerate([(ref,'Exact rectangle'),(contour-ref,'Polygon - rectangle')],1):
            fig,ax=plt.subplots(figsize=(6.3,4.7))
            im=ax.imshow(data,origin='lower',extent=[axis.min()/1000,axis.max()/1000]*2,cmap='viridis')
            fig.colorbar(im,ax=ax,label='mGal');ax.set_xlabel('x (km)');ax.set_ylabel('y (km)')
            ax.set_title('SYNTHETIC benchmark — '+label)
            fig.tight_layout();fig.savefig(out/f'FigS{i}.png',dpi=220);plt.close(fig)
    (out/'synthetic_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics


def benchmark_methods(x,y,z,rect,vertices,rho=350.,quadrature_order=8,repeats=3):
    """Timing requires repeats; results depend on hardware and vector sizes."""
    cases={'Nagy_rectangle':lambda:gz_rectangular(x,y,z,rect,rho),
           'Polygon_edges':lambda:gz_polygon(x,y,z,vertices,rect[4],rect[5],rho),
           'GLQ_volume':lambda:gz_rectangular_quadrature(x,y,z,rect,rho,order=quadrature_order)}
    reference=cases['Nagy_rectangle']()
    results=[]
    for name,func in cases.items():
        times=[];arr=None
        for i in range(repeats):
            t=time.perf_counter();arr=func();times.append(time.perf_counter()-t)
        results.append({'method':name,'median_seconds':float(np.median(times)),
                        **compare_fields(arr,reference)})
    return results


# -----------------------------------------------------------------------------
# Reproducible REAL-DATA pipeline. Never silently invent missing Marajo data.
# -----------------------------------------------------------------------------
def _read_surface(path, x_name='x_m', y_name='y_m', z_name='z_m'):
    a=np.genfromtxt(path,delimiter=',',names=True,encoding=None)
    if a.size==0:raise ValueError(f'Empty surface: {path}')
    return tuple(np.atleast_1d(a[k]).astype(float) for k in [x_name,y_name,z_name])


def _interpolate_surface(sx,sy,sv,x,y,method='linear'):
    from scipy.interpolate import griddata
    points=np.column_stack([sx,sy]); target=np.column_stack([np.asarray(x).ravel(),np.asarray(y).ravel()])
    vals=griddata(points,np.asarray(sv,float),target,method=method)
    return vals.reshape(np.asarray(x).shape)


def make_surface_cells_from_polygons(polygons,topo_xyz,moho_xyz,densities,fractions=None,
                                     topography_input='elevation_up_m',moho_input='depth_down_m',
                                     interpolation='linear'):
    """Sample common physical surfaces at the centroid of each polygonal column.

    Surface approximation is piecewise constant per polygon. Polygon geometries
    can differ, but each method is sampled from identical input surfaces.
    """
    from shapely.geometry import Polygon
    tx,ty,tv=topo_xyz;mx,my,mv=moho_xyz
    if topography_input not in ('elevation_up_m','depth_down_m'):
        raise ValueError('topography_input explicit elevation_up_m or depth_down_m')
    if moho_input not in ('elevation_up_m','depth_down_m'):
        raise ValueError('moho_input explicit elevation_up_m or depth_down_m')
    dens=np.atleast_1d(densities).astype(float)
    fr=np.linspace(0,1,len(dens)+1) if fractions is None else np.asarray(fractions,float)
    if len(fr)!=len(dens)+1 or abs(fr[0])>1.e-12 or abs(fr[-1]-1)>1.e-12 or np.any(np.diff(fr)<=0):
        raise ValueError('invalid layer fractions')
    cs=[]
    for v in polygons:
        poly=Polygon(_ccw_vertices(v))
        if not poly.is_valid or poly.area<=0:raise ValueError('invalid polygon')
        xx,yy=poly.centroid.x,poly.centroid.y
        ztop=float(_interpolate_surface(tx,ty,tv,np.array([xx]),np.array([yy]),interpolation)[0])
        zmoho=float(_interpolate_surface(mx,my,mv,np.array([xx]),np.array([yy]),interpolation)[0])
        if not np.isfinite([ztop,zmoho]).all():continue
        if topography_input=='elevation_up_m':ztop=-ztop
        if moho_input=='elevation_up_m':zmoho=-zmoho
        if zmoho<=ztop:continue
        for j,rho in enumerate(dens):
            cs.append(Cell(np.asarray(v,float),ztop+(zmoho-ztop)*fr[j],
                            ztop+(zmoho-ztop)*fr[j+1],float(rho)))
    if not cs:raise ValueError('No valid model cells. Check CRS, depth convention and surface overlap.')
    return cs


def _grid_rectangle_polygons(xmin,xmax,ymin,ymax,dx,dy):
    from shapely.geometry import box
    pieces=[]
    for x0 in np.arange(xmin,xmax-1e-9,dx):
        for y0 in np.arange(ymin,ymax-1e-9,dy):
            b=box(x0,y0,min(x0+dx,xmax),min(y0+dy,ymax))
            pieces.append(np.asarray(b.exterior.coords[:-1],float))
    return pieces


def _hex_polygon_footprints(xmin,xmax,ymin,ymax,radius):
    from shapely.geometry import Polygon,box
    boundary=box(xmin,ymin,xmax,ymax)
    dy=np.sqrt(3)*radius;dx=1.5*radius
    pieces=[]
    for i in range(-2,int(np.ceil((xmax-xmin)/dx))+3):
        cx=xmin+i*dx
        for j in range(-2,int(np.ceil((ymax-ymin)/dy))+3):
            cy=ymin+j*dy+(i%2)*dy/2
            inter=Polygon(regular_polygon(cx,cy,radius,6)).intersection(boundary)
            if inter.is_empty:continue
            polys=[inter] if inter.geom_type=='Polygon' else list(getattr(inter,'geoms',[]))
            for pol in polys:
                if pol.geom_type=='Polygon' and pol.area>1.e-8:
                    pieces.append(np.asarray(pol.exterior.coords[:-1],float))
    return pieces


def _voronoi_polygon_footprints(xmin,xmax,ymin,ymax,spacing,seed=431):
    """Repeatable jittered Voronoi mesh, clipped to rectangle; no hidden random state."""
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon,box
    xs=np.arange(xmin+spacing/2,xmax,spacing)
    ys=np.arange(ymin+spacing/2,ymax,spacing)
    if len(xs)<2 or len(ys)<2:raise ValueError('Voronoi requires >=2 sites along x and y')
    xx,yy=np.meshgrid(xs,ys); p=np.column_stack([xx.ravel(),yy.ravel()])
    rng=np.random.default_rng(seed)
    p+=rng.uniform(-.17*spacing,.17*spacing,p.shape)
    # Mirror seeds around boundaries: their Voronoi cells are finite inside bbox.
    mirrors=[p,np.column_stack((2*xmin-p[:,0],p[:,1])),
             np.column_stack((2*xmax-p[:,0],p[:,1])),
             np.column_stack((p[:,0],2*ymin-p[:,1])),
             np.column_stack((p[:,0],2*ymax-p[:,1]))]
    vor=Voronoi(np.vstack(mirrors)); boundary=box(xmin,ymin,xmax,ymax)
    polygons=[]
    for point_idx in range(len(p)):
        region=vor.regions[vor.point_region[point_idx]]
        if not region or -1 in region:continue
        inter=Polygon(vor.vertices[region]).intersection(boundary)
        if inter.is_empty:continue
        if inter.geom_type=='Polygon' and inter.area>1.e-8:
            polygons.append(np.asarray(inter.exterior.coords[:-1],float))
    return polygons



def save_comparison_figure(path,x,y,observed,regional,residual,title='Model comparison'):
    """Three-panel study figure for scattered observations, with explicit mGal labels.

    Saves both publication-ready 600 dpi TIFF and quick-look PNG, but color
    choices/layout still require journal layout and geologic map adaptation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xx=np.asarray(x,float).ravel()/1000
    yy=np.asarray(y,float).ravel()/1000
    fields=[np.asarray(k,float).ravel() for k in (observed,regional,residual)]
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,data,heading in zip(axs,fields,['Observed Bouguer','Modeled field','Difference (observed - model)']):
        plot=ax.scatter(xx,yy,c=data,s=34,cmap='viridis',edgecolors='none')
        cb=fig.colorbar(plot,ax=ax,shrink=.8);cb.set_label('mGal')
        ax.set_title(heading,fontsize=10)
        ax.set_xlabel('Projected x (km)');ax.set_ylabel('Projected y (km)')
        ax.set_aspect('equal',adjustable='box')
    fig.suptitle(title,fontsize=10)
    filepath=Path(path);filepath.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filepath.with_suffix('.png'),dpi=220)
    fig.savefig(filepath.with_suffix('.tif'),dpi=600)
    plt.close(fig)

def run_real_case(config_file):
    """Load user-provided Marajo CSV data; save method-wise modeled regional/residual.

    Requires config with REAL source paths. Does not reinterpret relative densities
    or observed anomaly corrections; consistency is investigator responsibility.
    Each geometry receives its OWN polygon mesh, clipped to the SAME bbox.
    """
    file=Path(config_file).resolve()
    config=json.loads(file.read_text(encoding='utf-8'))
    basedir=file.parent
    def getpath(key):
        value=config.get(key)
        if not value or 'FILL' in str(value):
            raise ValueError(f'Missing real input {key}; edit configuration before running')
        f=Path(value)
        f=(basedir/f).resolve() if not f.is_absolute() else f
        if not f.exists():raise FileNotFoundError(f)
        return f
    obs_path=getpath('bouguer_observations_csv')
    topo_path=getpath('topography_csv')
    moho_path=getpath('moho_csv')
    sx,sy,sz,bouguer=load_xyz_csv(obs_path)
    if not np.isfinite([sx,sy,sz,bouguer]).all():raise ValueError('invalid observed data; clean NaNs explicitly')
    if len(sx)==0:raise ValueError('no observations')
    topo=_read_surface(topo_path);moho=_read_surface(moho_path)
    bbox=config['bbox_m'];xmin,xmax,ymin,ymax=map(float,bbox)
    if not (xmin<xmax and ymin<ymax):raise ValueError('invalid bbox_m')
    spacing=float(config['horizontal_spacing_m'])
    if spacing<=0:raise ValueError('positive spacing required')
    contrast=config['density_contrasts_kgm3']
    if not isinstance(contrast,list) or not contrast or not np.isfinite(contrast).all():
        raise ValueError('density_contrasts_kgm3 must be explicit finite list')
    f=config.get('layer_fractions',None)
    polygons={
        'quadrangular':_grid_rectangle_polygons(xmin,xmax,ymin,ymax,spacing,spacing),
        'hexagonal':_hex_polygon_footprints(xmin,xmax,ymin,ymax,spacing/np.sqrt(3)),
        'polygonal_voronoi':_voronoi_polygon_footprints(xmin,xmax,ymin,ymax,spacing),
    }
    outpath=Path(config.get('output_directory','outputs/real_marajo'))
    outpath=(basedir/outpath).resolve() if not outpath.is_absolute() else outpath
    outpath.mkdir(parents=True,exist_ok=True)
    summary={'status':'MODELED_FROM_USER_DATA','n_observations':len(sx),'crs':config['crs'],
             'input_files':{k:str(getpath(k)) for k in ('bouguer_observations_csv','topography_csv','moho_csv')},
             'methods':{},'method_note':'Polygonal steps are centroid sampled, not inclined-face polyhedra.',
             'model_basis':config.get('regional_model_justification','NOT PROVIDED')}
    for name,footprints in polygons.items():
        cells=make_surface_cells_from_polygons(footprints,topo,moho,contrast,f,
                      config['topography_input'],config['moho_input'],
                      config.get('interpolation','linear'))
        t=time.perf_counter()
        if name=='quadrangular':
            reg=gz_cells(sx,sy,sz,cells,method='rectangular')
        else:
            reg=gz_cells(sx,sy,sz,cells,method='polygon')
        elapsed=time.perf_counter()-t
        residual=regional_residual(bouguer,reg)
        save_xyz_csv(outpath/f'{name}_regional_residual.csv',sx,sy,sz,
                     {'observed_bouguer_mGal':bouguer,'regional_mGal':reg,'residual_mGal':residual})
        if config.get('make_figures',True):
            save_comparison_figure(outpath/f'Fig_{name}',sx,sy,bouguer,reg,residual,
                                  title=('SYNTHETIC demonstration' if 'SYNTHETIC' in str(config['crs'])
                                         else 'Modeled gravity — '+name))
        summary['methods'][name]={'n_cells':len(cells),'total_volume_m3':float(sum(c.volume for c in cells)),
                                  'seconds':elapsed,'regional_min_mGal':float(np.min(reg)),
                                  'regional_max_mGal':float(np.max(reg)),
                                  'residual_rmse_from_zero_mGal':float(np.sqrt(np.mean(residual**2)))}
        if name=='quadrangular': ref=reg.copy()
        summary['methods'][name]['vs_quadrangular']=compare_fields(reg,ref)
    (outpath/'run_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary

# Compatibility entry point for existing regular prism model workflow.
def my_3Dgz(x,y,z,xprism,yprism,top,bottom,deltax,deltay,density):
    """Legacy API, rho in g/cm3; coordinates in metres; output mGal.

    Consistent with the original prism(3).py center-based interface, but uses
    this module's exact rectangular operator. Density may be scalar or per cell.
    """
    xx=np.asarray(xprism,float).ravel();yy=np.asarray(yprism,float).ravel()
    tt=np.broadcast_to(np.asarray(top,float),xx.shape).ravel()
    bb=np.broadcast_to(np.asarray(bottom,float),xx.shape).ravel()
    rho=np.broadcast_to(np.asarray(density,float),xx.shape).ravel()
    if not(len(xx)==len(yy)==len(tt)==len(bb)==len(rho)):raise ValueError('input sizes mismatch')
    ox,oy,oz=_observations(x,y,z); total=np.zeros_like(ox,float)
    for a,b,t,d,rr in zip(xx,yy,tt,bb,rho):
        if d>t:
            total+=gz_rectangular(ox,oy,oz,[a-deltax/2,a+deltax/2,b-deltay/2,b+deltay/2,t,d],rr,'g/cm3')
    return total


def main():
    import argparse
    p=argparse.ArgumentParser(description='Marajo reproducible gravity toolkit')
    p.add_argument('--demo',action='store_true',help='run synthetic experiment')
    p.add_argument('--real-config',default=None,help='JSON configuration for real Marajo data')
    p.add_argument('--output',default='outputs/synthetic',help='synthetic output directory')
    args=p.parse_args()
    if args.demo:
        print(json.dumps(synthetic_experiment(args.output),indent=2))
    elif args.real_config:
        print(json.dumps(run_real_case(args.real_config),indent=2,ensure_ascii=False))
    else:p.print_help()


if __name__=='__main__':main()

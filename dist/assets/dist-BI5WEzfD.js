import{$a as e,$o as t,$s as n,A as r,Aa as i,Ao as a,As as o,Ba as s,Bo as c,Bs as l,C as u,Ca as d,Cc as f,Ci as p,Co as m,Cs as h,D as g,Da as _,Do as v,Ds as y,E as b,Ea as x,Ec as S,Eo as C,Es as w,Fa as T,Fo as E,Fs as D,Gi as O,Gs as k,Ha as A,Ho as j,Hs as M,Ia as ee,Io as N,Is as P,Ja as F,Js as te,Ka as ne,Ki as re,Ko as ie,Ks as ae,La as oe,Ls as I,Ma as L,Mo as se,Mr as R,Ms as ce,Na as le,No as z,Ns as ue,O as de,Oa as fe,Oo as B,Os as pe,Pa as me,Po as he,Ps as ge,Qa as _e,Qi as ve,Qo as ye,Qs as be,Ra as V,Ro as xe,Rs as Se,S as Ce,Sa as we,Sc as Te,Si as Ee,So as De,Sr as Oe,Ss as H,T as ke,Ta as Ae,Ts as je,Ua as Me,Uo as Ne,Us as Pe,Va as Fe,Vo as Ie,Vs as Le,Wa as Re,Wo as ze,Ws as Be,Xa as Ve,Xi as He,Xo as U,Ya as Ue,Yo as We,Za as Ge,_ as Ke,_a as qe,_o as Je,_r as W,_s as Ye,a as Xe,ac as Ze,ao as Qe,as as $e,b as et,ba as tt,bc as G,bi as nt,bo as rt,br as it,bs as at,c as ot,ca as st,cc as ct,co as lt,cs as ut,d as dt,da as ft,dc as pt,do as mt,ds as ht,ea as gt,ec as _t,eo as vt,es as yt,f as bt,fa as xt,fc as St,fo as Ct,fs as wt,g as Tt,gc as Et,go as Dt,gs as Ot,h as kt,ha as At,hc as jt,ho as Mt,hs as Nt,i as Pt,ic as Ft,io as It,is as Lt,j as Rt,ja as zt,js as Bt,k as Vt,ka as Ht,ko as Ut,l as Wt,la as Gt,lo as Kt,ls as qt,m as Jt,ma as Yt,mc as Xt,mo as Zt,ms as Qt,n as $t,na as en,nc as tn,no as nn,ns as rn,o as an,oc as on,p as sn,pc as K,po as cn,ps as ln,qa as un,qo as dn,qs as fn,r as pn,rc as mn,rs as hn,s as gn,sa as _n,sc as vn,ss as yn,st as bn,tc as xn,to as Sn,ts as Cn,u as wn,ua as Tn,uc as q,uo as En,us as Dn,v as On,vc as kn,vi as An,vo as jn,vr as Mn,vs as Nn,w as Pn,wa as Fn,wc as In,wi as Ln,wo as Rn,ws as zn,x as Bn,xa as Vn,xi as Hn,xo as Un,xr as Wn,xs as Gn,y as Kn,yc as qn,yi as Jn,yo as Yn,yr as Xn,ys as Zn,za as Qn,zo as $n,zs as er}from"./dist-BUAIDZdM.js";
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var J;(function(e){e[e.float32=0]=`float32`,e[e.int32=1]=`int32`,e[e.bool=2]=`bool`,e[e.string=3]=`string`,e[e.complex64=4]=`complex64`})(J||={});var tr;(function(e){e[e.linear=0]=`linear`,e[e.relu=1]=`relu`,e[e.relu6=2]=`relu6`,e[e.prelu=3]=`prelu`,e[e.leakyrelu=4]=`leakyrelu`,e[e.sigmoid=5]=`sigmoid`,e[e.elu=6]=`elu`})(tr||={});
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let nr;function rr(e){nr=e.wasm.cwrap(vn,null,[`number`,`array`,`number`,`number`,`array`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ir(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t;if(i.dtype!==`float32`||a.dtype!==`float32`)throw Error(`_FusedMatMul for non non-float32 tensors not yet supported.`);let{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r,f=n.dataIdMap.get(i.dataId).id,p=n.dataIdMap.get(a.dataId).id,m=0;if(o!=null){let e=n.dataIdMap.get(o.dataId);if(e.shape.length!==1)throw Error(`_FusedMatMul only supports rank-1 bias but got rank ${e.shape.length}.`);m=e.id}let h=s==null?0:n.dataIdMap.get(s.dataId).id,g=tr[u];if(g==null)throw Error(`${u} activation not yet supported for FusedConv2D in the wasm backend.`);let _=c?i.shape[2]:i.shape[1],v=l?a.shape[1]:a.shape[2],y=R(i.shape.slice(0,-2),a.shape.slice(0,-2)),b=n.makeOutput([...y,_,v],i.dtype),x=n.dataIdMap.get(b.dataId).id,S=new Uint8Array(new Int32Array(i.shape).buffer),C=new Uint8Array(new Int32Array(a.shape).buffer);return nr(f,S,i.shape.length,p,C,a.shape.length,c,l,g,m,h,d||0,x),b}const ar={kernelName:vn,backendName:`wasm`,setupFunc:rr,kernelFunc:ir};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Y(e,t){let n;function r(t){n=t.wasm.cwrap(e,null,[`number`,`number`,`number`])}function i(e){let{backend:r,inputs:{x:i}}=e,a=r.dataIdMap.get(i.dataId).id,o=r.makeOutput(i.shape,t||i.dtype),s=r.dataIdMap.get(o.dataId).id;return G(o.shape)===0||n(a,J[i.dtype],s),o}return{kernelName:e,backendName:`wasm`,setupFunc:r,kernelFunc:i}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
const or=Y(`Abs`),sr=Y(Yt),cr=Y(At)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function X(e,t,n){let r;function i(t){r=t.wasm.cwrap(e,null,[`number`,`array`,`number`,`number`,`array`,`number`,`number`,`number`])}function a(e){let{backend:t,inputs:i}=e,{a,b:o}=i,s=t.dataIdMap.get(a.dataId).id,c=t.dataIdMap.get(o.dataId).id,l=n??a.dtype,u=R(a.shape,o.shape),d=t.makeOutput(u,l);if(G(u)===0)return d;let f=new Uint8Array(new Int32Array(a.shape).buffer),p=new Uint8Array(new Int32Array(o.shape).buffer),m=t.dataIdMap.get(d.dataId).id;return r(s,f,a.shape.length,c,p,o.shape.length,J[a.dtype],m),d}return{kernelName:e,backendName:`wasm`,setupFunc:i,kernelFunc:a}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
const lr=X(`Add`,!0);
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ur;function dr(e){ur=e.wasm.cwrap(qe,null,[`array`,`number`,`number`,`number`])}function fr(e){let{inputs:t,backend:n}=e,r=n.makeOutput(t[0].shape,t[0].dtype);if(G(r.shape)===0)return r;let i=t.map(e=>n.dataIdMap.get(e.dataId).id),a=new Uint8Array(new Int32Array(i).buffer),o=n.dataIdMap.get(r.dataId).id;return ur(a,i.length,J[r.dtype],o),r}const pr={kernelName:qe,backendName:`wasm`,setupFunc:dr,kernelFunc:fr};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function mr(e){let{inputs:{x:t},backend:n}=e;if(t.dtype===`string`)return en(n.readSync(t.dataId),t.shape,t.dtype);let r=n.makeOutput(t.shape,t.dtype),i=n.typedArrayFromHeap(t);return n.typedArrayFromHeap(r).set(i),r}const hr={kernelName:Un,backendName:`wasm`,kernelFunc:mr};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let gr;function _r(e){gr=e.wasm.cwrap(mn,null,[`number`,`array`,`number`,`number`,`number`,`array`,`number`])}function Z(e){let{inputs:t,backend:n,attrs:r}=e,[i,a]=yr(t.x.shape,r.perm),o=!0;for(let e=0;e<a.length;e++)a[e]!==e&&(o=!1);let s=vr(t.x.shape,r.perm),c={dataId:t.x.dataId,shape:i,dtype:t.x.dtype};if(o){let e=mr({inputs:t,backend:n});return e.shape=s,e}let l=n.makeOutput(s,c.dtype),u=n.dataIdMap.get(c.dataId).id,d=n.dataIdMap.get(l.dataId).id,f=new Uint8Array(new Int32Array(a).buffer),p=new Uint8Array(new Int32Array(c.shape).buffer);return gr(u,p,c.shape.length,J[c.dtype],d,f,a.length),l}function vr(e,t){let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];return n}function yr(e,t){let n=[],r=[];for(let i=0;i<e.length;++i)e[i]!==1&&n.push(e[i]),e[t[i]]!==1&&r.push(t[i]);for(let e=0;e<r.length;++e){let t=-1;for(let n=0;n<r.length;++n)r[n]>=e&&(t===-1||r[t]>r[n])&&(t=n);r[t]=e}return[n,r]}const br={kernelName:mn,backendName:`wasm`,kernelFunc:Z,setupFunc:_r};
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xr(e,t,n){let r=e.shape,i=e.shape.length,a=qn(t,r),o=a,s=it(o,i),c=null,l=!1;if(s!=null){let t=Array(i);for(let e=0;e<t.length;e++)t[e]=r[s[e]];o=Wn(o.length,i),c=Z({inputs:{x:e},attrs:{perm:s},backend:n});let a=n.dataIdMap.get(e.dataId).id;n.dataIdMap.get(c.dataId).id!==a&&(l=!0)}return{transposed:c,originalAxes:a,axes:o,inputWasTransposed:l}}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Sr;function Cr(e){Sr=e.wasm.cwrap(`All`,null,[`number, number, number`])}function wr(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=o,{transposed:l,axes:u,originalAxes:d,inputWasTransposed:f}=xr(o,i,t);if(f){let e=t.dataIdMap.get(l.dataId).id;c=l,s=e}let p=c.shape.length;W(`all`,u,p);let[m,h]=Mn(c.shape,u),g=G(h),_=t.makeOutput(m,o.dtype);if(G(c.shape)!==0){let e=t.dataIdMap.get(_.dataId).id;Sr(s,g,e)}return f&&t.disposeData(l.dataId),a&&(_.shape=Xn(_.shape,d)),_}const Tr={kernelName:`All`,backendName:`wasm`,setupFunc:Cr,kernelFunc:wr};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Er;function Dr(e){Er=e.wasm.cwrap(`Any`,null,[`number, number, number`])}function Or(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=o,{transposed:l,axes:u,originalAxes:d,inputWasTransposed:f}=xr(o,i,t);if(f){let e=t.dataIdMap.get(l.dataId).id;c=l,s=e}let p=c.shape.length;W(`any`,u,p);let[m,h]=Mn(c.shape,u),g=G(h),_=t.makeOutput(m,o.dtype);if(G(c.shape)!==0){let e=t.dataIdMap.get(_.dataId).id;Er(s,g,e)}return f&&t.disposeData(l.dataId),a&&(_.shape=Xn(_.shape,d)),_}const kr={kernelName:`Any`,backendName:`wasm`,setupFunc:Dr,kernelFunc:Or};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ar(e){let t;function n(n){t=n.wasm.cwrap(e,null,[`number`,`number`,`number`,`number`,`number`])}function r(e){let{backend:n,inputs:r,attrs:i}=e,{axis:a}=i,{x:o}=r,s=n.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,inputWasTransposed:f}=xr(o,a,n);if(f){let e=n.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e)}let p=l.shape.slice(0,-1),m=n.makeOutput(p,`int32`),h=n.dataIdMap.get(m.dataId).id,g=G(m.shape),_=l.shape[d[0]];return t(c,J[l.dtype],g,_,h),f&&n.disposeData(u.dataId),m}return{kernelName:e,backendName:`wasm`,setupFunc:n,kernelFunc:r}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
const jr=Ar(tt),Mr=Ar(Vn),Nr=Y(we),Pr=Y(d),Fr=Y(Fn),Ir=X(Ae,!1),Lr=Y(x)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Rr;function zr(e){Rr=e.wasm.cwrap(_,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Br(e){let{inputs:t,attrs:n,backend:r}=e,i=t.x,a=r.dataIdMap.get(i.dataId).id,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=n,u=Hn(i.shape,o,s,1,c,l),d=u.filterHeight,f=u.filterWidth,p=u.padInfo.top,m=u.padInfo.right,h=u.padInfo.bottom,g=u.padInfo.left,_=u.strideHeight,v=u.strideWidth,y=u.inChannels;if(u.dataFormat!==`channelsLast`)throw Error(`wasm backend does not support dataFormat:'${u.dataFormat}'. Please use 'channelsLast'.`);if(u.dilationWidth!==1||u.dilationHeight!==1)throw Error(`was backend only supports average pooling with dilation = [1, 1], got [${u.dilationHeight}, ${u.dilationWidth}].`);let b=r.makeOutput(u.outShape,`float32`),x=r.dataIdMap.get(b.dataId).id;return Rr(a,i.shape[0],i.shape[1],i.shape[2],d,f,p,m,h,g,_,v,y,x),b}const Vr={kernelName:_,backendName:`wasm`,setupFunc:zr,kernelFunc:Br};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Hr;function Ur(e){Hr=e.wasm.cwrap(`AvgPool3D`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Wr(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=Ee(i.shape,a,o,1,s,c,l),d=n.makeOutput(u.outShape,i.dtype);return Hr(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left),d}const Gr={kernelName:fe,backendName:`wasm`,setupFunc:Ur,kernelFunc:Wr};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Kr;function qr(e){Kr=e.wasm.cwrap(`AvgPool3DGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Jr(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r,u=Ee(a.shape,o,s,1,c,l),d=n.makeOutput(a.shape,a.dtype);return Kr(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left,u.filterDepth,u.filterHeight,u.filterWidth),d}const Yr={kernelName:Ht,backendName:`wasm`,setupFunc:qr,kernelFunc:Jr};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Xr;function Zr(e){Xr=e.wasm.cwrap(`AvgPoolGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Qr(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c}=r,l=Hn(a.shape,o,s,1,c),u=n.makeOutput(a.shape,a.dtype);return Xr(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inChannels,l.inHeight,l.inWidth,l.outHeight,l.outWidth,l.strideHeight,l.strideWidth,l.dilationHeight,l.dilationWidth,l.effectiveFilterHeight,l.effectiveFilterWidth,l.padInfo.top,l.padInfo.left,l.filterHeight,l.filterWidth),u}const $r={kernelName:i,backendName:`wasm`,setupFunc:Zr,kernelFunc:Qr};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Q(e){let{inputs:t,attrs:n}=e,{x:r}=t,{shape:i}=n,a=G(r.shape),o=Et(i,a);return q(a===G(o),()=>`new shape: ${o}, old shape: ${r.shape}. New shape and old shape must have the same number of elements.`),e.backend.incRef(r.dataId),{dataId:r.dataId,shape:o,dtype:r.dtype}}const ei={kernelName:Qt,backendName:`wasm`,kernelFunc:Q};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ti;function ni(e){ti=e.wasm.cwrap(zt,null,[`number`,`array`,`number`,`number`,`array`,`number`,`number`,`number`,`number`])}function ri(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;if(i.dtype!==`float32`||a.dtype!==`float32`)throw Error(`BatchMatMul for non non-float32 tensors not yet supported.`);let c=i.shape.length,l=a.shape.length,u=o?i.shape[c-2]:i.shape[c-1],d=s?a.shape[l-1]:a.shape[l-2],f=o?i.shape[c-1]:i.shape[c-2],p=s?a.shape[l-2]:a.shape[l-1],m=i.shape.slice(0,-2),h=a.shape.slice(0,-2),g=G(m),_=G(h),v=R(i.shape.slice(0,-2),a.shape.slice(0,-2)).concat([f,p]);q(u===d,()=>`Error in matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${i.shape} and ${a.shape} and transposeA=${o} and transposeB=${s} must match.`);let y=o?[g,u,f]:[g,f,u],b=s?[_,p,d]:[_,d,p],x=Q({inputs:{x:i},backend:n,attrs:{shape:y}}),S=Q({inputs:{x:a},backend:n,attrs:{shape:b}}),C=n.dataIdMap.get(x.dataId).id,w=n.dataIdMap.get(S.dataId).id,T=o?x.shape[2]:x.shape[1],E=s?S.shape[1]:S.shape[2],D=Math.max(g,_),O=n.makeOutput([D,T,E],x.dtype),k=n.dataIdMap.get(O.dataId).id,A=new Uint8Array(new Int32Array(x.shape).buffer),j=new Uint8Array(new Int32Array(S.shape).buffer);return ti(C,A,x.shape.length,w,j,S.shape.length,o,s,k),n.disposeData(x.dataId),n.disposeData(S.dataId),O.shape=v,O}const ii={kernelName:zt,backendName:`wasm`,setupFunc:ni,kernelFunc:ri};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ai(e,t,n,r){let i=Xt(n,G(t));if(r&&n!==`string`){let t=0;e.forEach(e=>{let n=G(e.shape);i.set(e.vals,t),t+=n})}else{let r=0;e.forEach(e=>{let a=n===`string`?pn(e.vals):e.vals,o=0;for(let n=0;n<e.shape[0];++n){let s=n*t[1]+r;for(let t=0;t<e.shape[1];++t)i[s+t]=a[o++]}r+=e.shape[1]})}return i}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function oi(e,t,n,r){if(e===t||e<t&&n<0||t<e&&n>1)return kn(0,r);let i=Math.abs(Math.ceil((t-e)/n)),a=kn(i,r);t<e&&n===1&&(n=-1),a[0]=e;for(let e=1;e<a.length;e++)a[e]=a[e-1]+n;return a}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function si(e,t,n,r,i){let a=de(r,t,n),o=G(n),s=K(r);if(a){let n=b(t,s);return i===`string`?e.slice(n,n+o):e.subarray(n,n+o)}let c=i===`string`?pn(e):e,l=O(r,i,c),u=O(n,i);for(let e=0;e<u.size;++e){let n=u.indexToLoc(e),r=n.map((e,n)=>e+t[n]);u.set(l.get(...r),...n)}return i===`string`?$t(u.values):u.values}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var ci=class{constructor(e,t,n,r,i,a){this.separator=st(e),this.nGramWidths=t,this.leftPad=st(n),this.rightPad=st(r),this.padWidth=i,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){let n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,a){for(let o=0;o<i;++o){let s=this.getPadWidth(a),c=Math.max(0,s-o),l=Math.max(0,s-(i-(o+1))),u=a-(c+l),d=t+(c>0?0:o-s),f=0;f+=c*this.leftPad.length;for(let t=0;t<u;++t)f+=e[d+t].length;f+=l*this.rightPad.length;let p=c+l+u-1;f+=p*this.separator.length,n[r+o]=new Uint8Array(f);let m=n[r+o],h=0,g=e=>e.forEach(e=>m[h++]=e);for(let e=0;e<c;++e)g(this.leftPad),g(this.separator);for(let t=0;t<u-1;++t)g(e[d+t]),g(this.separator);if(u>0){g(e[d+u-1]);for(let e=0;e<l;++e)g(this.separator),g(this.rightPad)}else{for(let e=0;e<l-1;++e)g(this.rightPad),g(this.separator);g(this.rightPad)}}}compute(e,t){let n=e.length,r=t.length;if(r>0){let e=t[0];if(e!==0)throw Error(`First split value must be 0, got ${e}`);for(let i=1;i<r;++i){let r=t[i]>=e;if(r&&=t[i]<=n,!r)throw Error(`Invalid split value ${t[i]}, must be in [${e}, ${n}]`);e=t[i]}if(e!==n)throw Error(`Last split value must be data size. Expected ${n}, got ${e}`)}let i=r-1,a=Xt(`int32`,r);if(n===0||r===0){let e=Array(n);for(let e=0;e<=i;++e)a[e]=0;return[e,a]}a[0]=0;for(let e=1;e<=i;++e){let n=t[e]-t[e-1],r=0;this.nGramWidths.forEach(e=>{r+=this.getNumNGrams(n,e)}),this.preserveShort&&n>0&&r===0&&(r=1),a[e]=a[e-1]+r}let o=Array(a[i]);for(let n=0;n<i;++n){let r=t[n],i=a[n];if(this.nGramWidths.forEach(a=>{let s=t[n+1]-t[n],c=this.getNumNGrams(s,a);this.createNGrams(e,r,o,i,c,a),i+=c}),this.preserveShort&&i===a[n]){let a=t[n+1]-t[n];if(a===0)continue;let s=a+2*this.padWidth;this.createNGrams(e,r,o,i,1,s)}}return[o,a]}};function li(e,t,n,r,i,a,o,s){return new ci(n,r,i,a,o,s).compute(e,t)}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ui(e,t,n,r){if(!e.length)return;if(t.length===0){for(let t=0;t<e.length;++t)r.push(e.subarray(t,t+1));return}if(t.length===1){let i=t[0],a=e.indexOf(i);for(;a!==-1;){let t=e.subarray(0,a);(!n||t.length!==0)&&r.push(t),e=e.subarray(a+1),a=e.indexOf(i)}(!n||e.length!==0)&&r.push(e);return}let i=0;for(let a=0;a<e.length+1;a++)if(a===e.length||t.indexOf(e[a])!==-1){let t=e.subarray(i,a);(!n||t.length!==0)&&r.push(t),i=a+1}}function di(e,t,n){let r=e.length,i=[],a=0,o=0,s=Array(r);for(let c=0;c<r;++c){let r=i.length;ui(e[c],t,n,i);let l=i.length-r;s[c]=l,a+=l,o=Math.max(o,l)}let c=Xt(`int32`,a*2),l=Array(a),u=[r,o],d=0;for(let e=0;e<r;++e)for(let t=0;t<s[e];++t)c[d*2]=e,c[d*2+1]=t,l[d]=i[d],++d;return[c,l,u]}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function fi(e,t){let n=Xt(`int32`,e.length);for(let r=0;r<e.length;++r)n[r]=ft(e[r]).modulo(t).getLowBitsUnsigned();return n}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function pi(e,t,n,r){let i=qn(t,n)[0],a=[1,n[0],1];for(let e=0;e<i;e++)a[0]*=n[e];a[1]=n[i];for(let e=i+1;e<n.length;e++)a[2]*=n[e];let o=/* @__PURE__ */ new Map,s=new Int32Array(n[i]),c=new _n(a,r,e),l=[],u=a[0]===1&&a[2]===1;for(let t=0;t<n[i];t++){let n;if(u)n=e[t].toString();else{let e=[];for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)e.push(c.get(n,t,r));n=e.join(`,`)}let r=o.get(n);if(r!=null)s[t]=r;else{let e=o.size;o.set(n,e),s[t]=e,l.push(t)}}let d=a.slice();d[1]=o.size;let f=new _n(d,r);l.forEach((e,t)=>{for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)f.set(c.get(n,e,r),n,t,r)});let p=n.slice();return p[i]=d[1],{outputValues:f.values,outputShape:p,indices:s}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function mi(e){let{inputs:{x:t},attrs:{begin:n,size:r},backend:i}=e,[a,o]=Vt(t,n,r),s=de(t.shape,a,o),c=i.readSync(t.dataId),l=i.makeOutput(o,t.dtype),u=K(t.shape),d=i.dataIdMap.get(l.dataId);if(s){let e=b(a,u);return t.dtype===`string`?d.stringBytes=c.slice(e,e+G(o)):i.typedArrayFromHeap(l).set(c.subarray(e,e+G(o))),l}if(t.dtype===`string`)return d.stringBytes=si(c,a,o,t.shape,t.dtype),l;let f=i.typedArrayFromHeap(l),p=t.shape.length;if(p===2)hi(c,u[0],f,a,o);else if(p===3)gi(c,u[0],u[1],f,a,o);else if(p===4)_i(c,u[0],u[1],u[2],f,a,o);else{let e=si(c,a,o,t.shape,t.dtype);f.set(e)}return l}function hi(e,t,n,r,i){let a=0,o=r[0],s=r[1],c=o+i[0];for(let r=o;r<c;r++){let o=r*t+s;n.set(e.subarray(o,o+i[1]),a),a+=i[1]}}function gi(e,t,n,r,i,a){let o=0,s=i[0],c=i[1],l=i[2],u=s+a[0],d=c+a[1];for(let i=s;i<u;i++)for(let s=c;s<d;s++){let c=i*t+s*n+l;r.set(e.subarray(c,c+a[2]),o),o+=a[2]}}function _i(e,t,n,r,i,a,o){let s=0,c=a[0],l=a[1],u=a[2],d=c+o[0],f=l+o[1],p=u+o[2],m=a[3];for(let a=c;a<d;a++)for(let c=l;c<f;c++)for(let l=u;l<p;l++){let u=a*t+c*n+l*r+m;i.set(e.subarray(u,u+o[3]),s),s+=o[3]}}const vi={kernelName:Bt,backendName:`wasm`,kernelFunc:mi};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function yi(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r,s=a.reduce((e,t)=>e*t),c=Kn(i.shape,a,s),l=On(c.length,a.length),u=et(i.shape,a,s),d=Bn(o,a.length),f=Ce(u,o,a.length),p=Q({inputs:{x:i},backend:n,attrs:{shape:c}}),m=Z({inputs:{x:p},backend:n,attrs:{perm:l}}),h=Q({inputs:{x:m},backend:n,attrs:{shape:u}}),g=mi({inputs:{x:h},backend:n,attrs:{begin:d,size:f}});return n.disposeData(p.dataId),n.disposeData(m.dataId),n.disposeData(h.dataId),g}const bi={kernelName:L,backendName:`wasm`,kernelFunc:yi};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let xi;function Si(e){xi=e.wasm.cwrap(le,null,[`number`,`number`,`boolean`,`number`,`number`,`number`])}function Ci(e){let{backend:t,inputs:n,attrs:r}=e,{x:i,weights:a}=n,{size:o}=r,s=a.shape.reduce((e,t)=>e*t,1)!==0,c=i.shape.length===1?[o]:[i.shape[0],o],l=t.makeOutput(c,a.dtype);function u(e){return t.dataIdMap.get(e.dataId).id}return xi(u(i),o,s,u(a),J[a.dtype],u(l)),l}const wi={kernelName:le,backendName:`wasm`,setupFunc:Si,kernelFunc:Ci},Ti=X(me,!0)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ei(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.typedArrayFromHeap(r),o=n.typedArrayFromHeap(i),s=R(Array.from(a),Array.from(o));return n.makeOutput([s.length],`int32`,void 0,new Int32Array(s))}const Di={kernelName:T,backendName:`wasm`,kernelFunc:Ei};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $(e){let{inputs:{x:t},attrs:{dtype:n},backend:r}=e,i=r.makeOutput(t.shape,n),a=r.typedArrayFromHeap(t);return r.typedArrayFromHeap(i).set(a),i}const Oi={kernelName:ee,backendName:`wasm`,kernelFunc:$},ki=Y(oe)
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ai;function ji(e){Ai=e.wasm.cwrap(V,null,[`number`,`number`,`number`,`number`])}function Mi(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{clipValueMin:a,clipValueMax:o}=r,s=n.dataIdMap.get(i.dataId).id,c=n.makeOutput(i.shape,i.dtype),l=n.dataIdMap.get(c.dataId).id;return Ai(s,a,o,l),c}const Ni={kernelName:V,backendName:`wasm`,setupFunc:ji,kernelFunc:Mi};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Pi(e){let{inputs:t,backend:n}=e,r=qn(e.attrs.axis,t[0].shape)[0],i=t.map(e=>e.shape);Pn(i,r);let a=ke(t.map(e=>e.shape),r),o=t.filter(e=>G(e.shape)>0);if(o.length===1)return mr({inputs:{x:o[0]},backend:n});let s=n.makeOutput(a,t[0].dtype);if(G(a)===0)return s;if(o[0].dtype===`string`){let e=o.map(e=>{let t=[-1,G(e.shape.slice(r))];return Q({inputs:{x:e},backend:n,attrs:{shape:t}})}),i=e.map(e=>({vals:n.readSync(e.dataId),shape:e.shape}));a=ke(e.map(e=>e.shape),1);let c=e[0].shape[0]===1,l=ai(i,a,t[0].dtype,c);s.shape=ke(o.map(e=>e.shape),r);let u=n.dataIdMap.get(s.dataId);return u.stringBytes=$t(l),e.forEach(e=>n.disposeData(e.dataId)),s}let c=G(o[0].shape.slice(0,r)),l=0,u=o.map(e=>{let t=G(e.shape.slice(r));return l+=t,t}),d=o.map(e=>n.typedArrayFromHeap(e)),f=n.typedArrayFromHeap(s);for(let e=0;e<c;e++){let t=e*l;for(let n=0;n<d.length;n++){let r=u[n],i=e*r,a=d[n].subarray(i,i+r);f.set(a,t),t+=r}}return s}const Fi={kernelName:Qn,backendName:`wasm`,kernelFunc:Pi};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ii;function Li(e){Ii=e.wasm.cwrap(s,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Ri(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a}=t,o=r.dataIdMap.get(i.dataId).id,s=r.dataIdMap.get(a.dataId).id,{strides:c,dilations:l,pad:u,dimRoundingMode:d,dataFormat:f}=n,m=p(f),h=An(i.shape,a.shape,c,l,u,d,!1,m),g=h.filterHeight,_=h.filterWidth,v=h.padInfo.top,y=h.padInfo.right,b=h.padInfo.bottom,x=h.padInfo.left,S=h.dilationHeight,C=h.dilationWidth,w=h.strideHeight,T=h.strideWidth,E=h.inChannels,D=h.outChannels,O=+(h.padInfo.type===`SAME`);if(h.dataFormat!==`channelsLast`)throw Error(`wasm backend Conv2D does not support dataFormat:'${h.dataFormat}'. Please use 'channelsLast'.`);let k=r.makeOutput(h.outShape,`float32`),A=r.dataIdMap.get(k.dataId).id;return Ii(o,i.shape[0],i.shape[1],i.shape[2],s,g,_,v,y,b,x,O,S,C,w,T,E,D,A),k}const zi={kernelName:s,backendName:`wasm`,setupFunc:Li,kernelFunc:Ri};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Bi;function Vi(e){Bi=e.wasm.cwrap(Fe,null,/* @__PURE__ */ `number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number`.split(`.`))}function Hi(e){let{backend:t,inputs:n,attrs:r}=e,{dy:i,filter:a}=n,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,inputShape:u}=r,d=p(c),f=An(u,a.shape,o,1,s,l,!1,d),{batchSize:m,filterHeight:h,filterWidth:g,inChannels:_,inHeight:v,inWidth:y,outChannels:b,outHeight:x,outWidth:S,strideHeight:C,strideWidth:w}=f,T=h-1-f.padInfo.top,E=g-1-f.padInfo.left,D=f.dataFormat===`channelsLast`,O=K(f.inShape),k=K(i.shape),[A,j,M]=K(a.shape),ee=O[0],N=D?O[1]:O[2],P=D?O[2]:1,F=D?1:O[1],te=k[0],ne=D?k[1]:k[2],re=D?k[2]:1,ie=D?1:k[1],ae=t.makeOutput(f.inShape,`float32`),oe=t.dataIdMap.get(ae.dataId).id,I=t.dataIdMap.get(i.dataId).id,L=t.dataIdMap.get(a.dataId).id;return Bi(I,L,m,h,g,v,y,_,x,S,b,C,w,T,E,A,j,M,ee,N,P,F,te,ne,re,ie,oe),ae}const Ui={kernelName:Fe,backendName:`wasm`,setupFunc:Vi,kernelFunc:Hi};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Wi;function Gi(e){Wi=e.wasm.cwrap(A,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Ki(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;if(i.dtype!==`float32`)throw Error(`Tensor x must have dtype float32, got ${i.dtype}`);if(a.dtype!==`float32`)throw Error(`Tensor filter must have dtype float32, got ${a.dtype}`);let l=Jn(i.shape,a.shape,o,c,s),u=n.makeOutput(l.outShape,i.dtype);return Wi(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inDepth,l.inHeight,l.inWidth,l.inChannels,l.outDepth,l.outHeight,l.outWidth,l.outChannels,l.strideDepth,l.strideHeight,l.strideWidth,l.dilationDepth,l.dilationHeight,l.dilationWidth,l.filterDepth,l.filterHeight,l.filterWidth,l.padInfo.front,l.padInfo.top,l.padInfo.left),u}const qi={kernelName:A,backendName:`wasm`,setupFunc:Gi,kernelFunc:Ki};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ji;function Yi(e){Ji=e.wasm.cwrap(Me,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Xi(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r;if(i.dtype!==`float32`)throw Error(`Tensor dy must have dtype float32, got ${i.dtype}`);if(a.dtype!==`float32`)throw Error(`Tensor filter must have dtype float32, got ${a.dtype}`);let l=Jn(i.shape,c,o,1,s),u=n.makeOutput(l.filterShape,a.dtype);return Ji(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inDepth,l.inHeight,l.inWidth,l.inChannels,l.outDepth,l.outHeight,l.outWidth,l.outChannels,l.strideDepth,l.strideHeight,l.strideWidth,l.dilationDepth,l.dilationHeight,l.dilationWidth,l.filterDepth,l.filterHeight,l.filterWidth,l.padInfo.front,l.padInfo.top,l.padInfo.left),u}const Zi={kernelName:Me,backendName:`wasm`,setupFunc:Yi,kernelFunc:Xi};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Qi;function $i(e){Qi=e.wasm.cwrap(Re,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ea(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r;if(i.dtype!==`float32`)throw Error(`Tensor dy must have dtype float32, got ${i.dtype}`);if(a.dtype!==`float32`)throw Error(`Tensor filter must have dtype float32, got ${a.dtype}`);let l=Jn(c,a.shape,s,1,o),u=n.makeOutput(l.inShape,i.dtype);return Qi(n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inDepth,l.inHeight,l.inWidth,l.inChannels,l.outDepth,l.outHeight,l.outWidth,l.outChannels,l.strideDepth,l.strideHeight,l.strideWidth,l.dilationDepth,l.dilationHeight,l.dilationWidth,l.filterDepth,l.filterHeight,l.filterWidth,l.padInfo.front,l.padInfo.top,l.padInfo.left),u}const ta={kernelName:Re,backendName:`wasm`,setupFunc:$i,kernelFunc:ea},na=Y(`Cos`),ra=Y(ne)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var ia;(function(e){e[e.bilinear=0]=`bilinear`,e[e.nearest=1]=`nearest`})(ia||={});let aa;function oa(e){aa=e.wasm.cwrap(un,null,[`number`,`number`,`number`,`number`,`array`,`number`,`number`,`number`,`number`,`number`])}function sa(e){let{backend:t,inputs:n,attrs:r}=e,{method:i,extrapolationValue:a,cropSize:o}=r,{image:s,boxes:c,boxInd:l}=n,u=c.shape[0],[d,f]=o,p=[u,d,f,s.shape[3]],m=t.dataIdMap.get(s.dataId),h;s.dtype!==`float32`&&(h=$({backend:t,inputs:{x:s},attrs:{dtype:`float32`}}),m=t.dataIdMap.get(h.dataId));let g=m.id,_=t.dataIdMap.get(c.dataId).id,v=t.dataIdMap.get(l.dataId).id,y=t.makeOutput(p,`float32`),b=t.dataIdMap.get(y.dataId).id,x=new Uint8Array(new Int32Array(s.shape).buffer);return aa(g,_,v,u,x,d,f,ia[i],a,b),h!=null&&t.disposeData(h.dataId),y}const ca={kernelName:un,backendName:`wasm`,setupFunc:oa,kernelFunc:sa};
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let la;function ua(e){la=e.wasm.cwrap(F,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function da(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r,c=i.shape.length;q(i.dtype===`float32`||i.dtype===`int32`,()=>`cumprod does not support ${i.dtype} tensors in the WASM backend`);let l=it([a],c),u=i;l!==null&&(u=Z({inputs:{x:i},attrs:{perm:l},backend:n}));let d=Wn(1,c)[0];W(`cumprod`,[d],c);let f=n.makeOutput(u.shape,u.dtype),p=u.shape[d],m=n.dataIdMap.get(u.dataId).id,h=n.dataIdMap.get(f.dataId).id;la(m,+!!o,+!!s,p,h,J[i.dtype]);let g=f;if(l!==null){let e=Oe(l);g=Z({inputs:{x:f},attrs:{perm:e},backend:n}),n.disposeData(u.dataId),n.disposeData(f.dataId)}return g}const fa={kernelName:F,backendName:`wasm`,setupFunc:ua,kernelFunc:da};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let pa;function ma(e){pa=e.wasm.cwrap(Ue,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function ha(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r,c=i.shape.length;q(i.dtype===`float32`||i.dtype===`int32`,()=>`cumsum does not support ${i.dtype} tensors in the WASM backend`);let l=it([a],c),u=i;l!==null&&(u=Z({inputs:{x:i},attrs:{perm:l},backend:n}));let d=Wn(1,c)[0];W(`cumsum`,[d],c);let f=n.makeOutput(u.shape,u.dtype),p=u.shape[d],m=n.dataIdMap.get(u.dataId).id,h=n.dataIdMap.get(f.dataId).id;pa(m,+!!o,+!!s,p,h,J[i.dtype]);let g=f;if(l!==null){let e=Oe(l);g=Z({inputs:{x:f},attrs:{perm:e},backend:n}),n.disposeData(u.dataId),n.disposeData(f.dataId)}return g}const ga={kernelName:Ue,backendName:`wasm`,setupFunc:ma,kernelFunc:ha};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let _a;function va(e){_a=e.wasm.cwrap(`DenseBincount`,null,[`number`,`array`,`number`,`number`,`boolean`,`number`,`number`,`boolean`,`number`])}function ya(e){let{backend:t,inputs:n,attrs:r}=e,{x:i,weights:a}=n,{size:o,binaryOutput:s}=r,c=a.shape.reduce((e,t)=>e*t,1)!==0,l=i.shape.length===1?[o]:[i.shape[0],o],u=t.makeOutput(l,a.dtype);function d(e){return t.dataIdMap.get(e.dataId).id}return _a(d(i),new Uint8Array(new Int32Array(i.shape).buffer),i.shape.length,o,c,d(a),J[a.dtype],s,d(u)),u}const ba={kernelName:Ve,backendName:`wasm`,setupFunc:va,kernelFunc:ya};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let xa;function Sa(e){xa=e.wasm.cwrap(Ge,null,[`number`,`number`,`number`,`array`,`number`,`array`,`array`,`number`,`number`])}function Ca(e){let{backend:t,inputs:n,attrs:r}=e,{x:i}=n,{blockSize:a,dataFormat:o}=r,s=i.shape[0],c=o===`NHWC`?i.shape[1]:i.shape[2],l=o===`NHWC`?i.shape[2]:i.shape[3],u=o===`NHWC`?i.shape[3]:i.shape[1],d=c*a,f=l*a,p=u/(a*a),m=o===`NHWC`?[s,d,f,p]:[s,p,d,f],h=t.makeOutput(m,`float32`),g=t.dataIdMap.get(i.dataId).id,_=new Uint8Array(new Int32Array(K(i.shape)).buffer),v=new Uint8Array(new Int32Array(m).buffer),y=new Uint8Array(new Int32Array(K(m)).buffer),b=t.dataIdMap.get(h.dataId).id;return xa(g,a,+(o===`NHWC`),_,i.shape.length-1,v,y,m.length,b),h}const wa={kernelName:Ge,backendName:`wasm`,setupFunc:Sa,kernelFunc:Ca};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ta;function Ea(e){Ta=e.wasm.cwrap(_e,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Da(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a}=t,o=r.dataIdMap.get(i.dataId).id,s=r.dataIdMap.get(a.dataId).id,{strides:c,dilations:l,pad:u,dimRoundingMode:d}=n,f=l??[1,1],p=An(i.shape,a.shape,c,f,u,d,!0),m=p.filterHeight,h=p.filterWidth,g=p.padInfo.top,_=p.padInfo.right,v=p.padInfo.bottom,y=p.padInfo.left,b=p.dilationHeight,x=p.dilationWidth,S=p.strideHeight,C=p.strideWidth,w=p.inChannels,T=p.outChannels,E=+(p.padInfo.type===`SAME`);if(p.dataFormat!==`channelsLast`)throw Error(`wasm backend DepthwiseConv2dNative does not support dataFormat:'${p.dataFormat}'. Please use 'channelsLast'.`);let D=r.makeOutput(p.outShape,`float32`),O=r.dataIdMap.get(D.dataId).id;return Ta(o,i.shape[0],i.shape[1],i.shape[2],s,m,h,g,_,v,y,E,b,x,S,C,w,T,O),D}const Oa={kernelName:_e,backendName:`wasm`,setupFunc:Ea,kernelFunc:Da};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ka;function Aa(e){ka=e.wasm.cwrap(`Diag`,null,[`number`,`number`,`number`,`number`])}function ja(e){let{inputs:t,backend:n}=e,{x:r}=t,i=G(r.shape),a=n.makeOutput([...r.shape,...r.shape],r.dtype);return ka(n.dataIdMap.get(r.dataId).id,J[r.dtype],i,n.dataIdMap.get(a.dataId).id),a}const Ma={kernelName:e,backendName:`wasm`,setupFunc:Aa,kernelFunc:ja};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Na;function Pa(e){Na=e.wasm.cwrap(vt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Fa(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;if(i.dtype!==a.dtype)throw Error(`Dilation2D error: x must have the same dtype as filter. Got ${i.dtype} and ${a.dtype}`);let l=nt(i.shape,a.shape,o,s,`NHWC`,c),u=n.makeOutput(l.outShape,i.dtype);return Na(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(u.dataId).id,J[i.dtype],l.batchSize,l.inChannels,l.inHeight,l.inWidth,l.outHeight,l.outWidth,l.strideHeight,l.strideWidth,l.dilationHeight,l.dilationWidth,l.filterHeight,l.filterWidth,l.padInfo.top,l.padInfo.left),u}const Ia={kernelName:vt,backendName:`wasm`,setupFunc:Pa,kernelFunc:Fa};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let La;function Ra(e){La=e.wasm.cwrap(Sn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function za(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,dy:o}=t,{strides:s,pad:c,dilations:l}=r;if(i.dtype!==a.dtype||i.dtype!==o.dtype)throw Error(`Dilation2DBackpropFilter error: x must have the same dtype as filter and dy. Got ${i.dtype}, ${a.dtype}, and ${o.dtype}`);let u=nt(i.shape,a.shape,s,c,`NHWC`,l),d=n.makeOutput(a.shape,a.dtype);return La(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(o.dataId).id,n.dataIdMap.get(d.dataId).id,J[i.dtype],u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.filterHeight,u.filterWidth,u.padInfo.top,u.padInfo.left),d}const Ba={kernelName:Sn,backendName:`wasm`,setupFunc:Ra,kernelFunc:za};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Va;function Ha(e){Va=e.wasm.cwrap(nn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Ua(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,dy:o}=t,{strides:s,pad:c,dilations:l}=r;if(i.dtype!==a.dtype||i.dtype!==o.dtype)throw Error(`Dilation2DBackpropInput error: x must have the same dtype as filter and dy. Got ${i.dtype}, ${a.dtype}, and ${o.dtype}`);let u=nt(i.shape,a.shape,s,c,`NHWC`,l),d=n.makeOutput(i.shape,i.dtype);return Va(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(o.dataId).id,n.dataIdMap.get(d.dataId).id,J[i.dtype],u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.filterHeight,u.filterWidth,u.padInfo.top,u.padInfo.left),d}const Wa={kernelName:nn,backendName:`wasm`,setupFunc:Ha,kernelFunc:Ua},Ga=Y(`Elu`)
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ka;function qa(e){Ka=e.wasm.cwrap(It,null,[`number`,`number`,`number`])}function Ja(e){let{inputs:t,backend:n}=e,{dy:r,y:i}=t,a=n.makeOutput(i.shape,`float32`),o=e=>n.dataIdMap.get(e.dataId).id;return Ka(o(i),o(r),o(a)),a}const Ya={kernelName:It,backendName:`wasm`,setupFunc:qa,kernelFunc:Ja},Xa=X(Qe,!1,`bool`),Za=Y(`Erf`),Qa=Y(`Exp`,`float32`)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $a(e){let{inputs:t,attrs:n,backend:r}=e,{input:i}=t,{dim:a}=n,o=i.shape.length,s=i.shape.slice(),c=a;return a<0&&(q(-(o+1)<=a,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+a+1),s.splice(c,0,1),Q({inputs:{x:i},backend:r,attrs:{shape:s}})}const eo={kernelName:lt,backendName:`wasm`,kernelFunc:$a},to=Y(Kt,`float32`)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function no(e){let{attrs:{shape:t,value:n},backend:r}=e,{attrs:{dtype:i}}=e;i||=jt(n);let a=r.makeOutput(t,i);return r.typedArrayFromHeap(a).fill(n),a}const ro={kernelName:En,backendName:`wasm`,kernelFunc:no};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let io;function ao(e){io=e.wasm.cwrap(mt,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function oo(e){let{inputs:t,backend:n}=e,{image:r}=t,i=n.makeOutput(r.shape,r.dtype),a=n.dataIdMap.get(r.dataId).id,o=n.dataIdMap.get(i.dataId).id,[s,c,l,u]=r.shape;return io(a,s,c,l,u,o),i}const so={kernelName:mt,backendName:`wasm`,kernelFunc:oo,setupFunc:ao},co=Y(Ct),lo=X(cn,!1)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let uo;function fo(e){uo=e.wasm.cwrap(Zt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function po(e){let{backend:t,inputs:n,attrs:r}=e,{varianceEpsilon:i}=r,{x:a,mean:o,variance:s,offset:c,scale:l}=n,u=t.dataIdMap.get(a.dataId).id,d=t.dataIdMap.get(o.dataId).id,f=t.dataIdMap.get(s.dataId).id,p=c==null?0:t.dataIdMap.get(c.dataId).id,m=l==null?0:t.dataIdMap.get(l.dataId).id,h=t.makeOutput(a.shape,a.dtype);if(G(a.shape)===0)return h;let g=t.dataIdMap.get(h.dataId).id;return uo(u,d,f,p,m,i,g),h}const mo={kernelName:Zt,backendName:`wasm`,setupFunc:fo,kernelFunc:po};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ho;function go(e){ho=e.wasm.cwrap(Mt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function _o(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dataFormat:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=n,h=An(i.shape,a.shape,c,u,l,f),g=tr[p];if(g==null)throw Error(`${p} activation not yet supported for FusedConv2D in the wasm backend.`);let _=r.dataIdMap.get(i.dataId).id,v=r.dataIdMap.get(a.dataId).id,y=h.outChannels,b=0;if(o!=null){let e=r.dataIdMap.get(o.dataId);if(e.shape.length!==1)throw Error(`FusedConv2D only supports rank-1 bias but got rank ${e.shape.length}.`);if(e.shape[0]!==y)throw Error(`FusedConv2D bias shape (${e.shape}) does not match the number of output channels (${y})`);b=e.id}let x=h.filterHeight,S=h.filterWidth,C=h.padInfo.top,w=h.padInfo.right,T=h.padInfo.bottom,E=h.padInfo.left,D=h.dilationHeight,O=h.dilationWidth,k=h.strideHeight,A=h.strideWidth,j=h.inChannels,M=+(h.padInfo.type===`SAME`),ee=h.batchSize,N=h.inHeight,P=h.inWidth;if(d!==`NHWC`)throw Error(`wasm backend FusedConv2D does not support dataFormat:'${d}'. Please use 'NHWC'.`);let F=r.makeOutput(h.outShape,`float32`),te=r.dataIdMap.get(F.dataId).id,ne=s==null?0:r.dataIdMap.get(s.dataId).id;return ho(_,ee,N,P,v,x,S,b,C,w,T,E,M,D,O,k,A,j,y,g,ne,m||0,te),F}const vo={kernelName:Mt,backendName:`wasm`,setupFunc:go,kernelFunc:_o};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let yo;function bo(e){yo=e.wasm.cwrap(Dt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function xo(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dataFormat:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=n,h=An(i.shape,a.shape,c,u,l,f,!0),g=tr[p];if(g==null)throw Error(`${p} activation not yet supported for FusedDepthwiseConv2D in the wasm backend.`);let _=r.dataIdMap.get(i.dataId).id,v=r.dataIdMap.get(a.dataId).id,y=h.outChannels,b=0;if(o!=null){let e=r.dataIdMap.get(o.dataId);if(e.shape.length!==1)throw Error(`FusedDepthwiseConv2D only supports rank-1 bias but got rank ${e.shape.length}.`);if(e.shape[0]!==y)throw Error(`FusedDepthwiseConv2D bias shape (${e.shape}) does not match the number of output channels (${y})`);b=e.id}let x=h.filterHeight,S=h.filterWidth,C=h.padInfo.top,w=h.padInfo.right,T=h.padInfo.bottom,E=h.padInfo.left,D=h.dilationHeight,O=h.dilationWidth,k=h.strideHeight,A=h.strideWidth,j=h.inChannels,M=+(h.padInfo.type===`SAME`),ee=h.batchSize,N=h.inHeight,P=h.inWidth;if(d!==`NHWC`)throw Error(`wasm backend FusedDepthwiseConv2D does not support dataFormat:'${d}'. Please use 'NHWC'.`);let F=r.makeOutput(h.outShape,`float32`),te=r.dataIdMap.get(F.dataId).id,ne=s==null?0:r.dataIdMap.get(s.dataId).id;return yo(_,ee,N,P,v,x,S,b,C,w,T,E,M,D,O,k,A,j,y,g,ne,m||0,te),F}const So={kernelName:Dt,backendName:`wasm`,setupFunc:bo,kernelFunc:xo};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Co;function wo(e){Co=e.wasm.cwrap(Je,null,[`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`])}function To(e){let{backend:t,inputs:n}=e,{params:r,indices:i}=n,[a,o,s,c]=Rt(r,i),l=t.makeOutput(a,r.dtype);if(o===0)return l;let u=i.shape,d=u[u.length-1],f=t.dataIdMap.get(r.dataId).id,p=t.dataIdMap.get(i.dataId).id,m=new Uint8Array(new Int32Array(c).buffer),h=t.dataIdMap.get(l.dataId).id;return Co(f,J[r.dtype],p,o,d,s,m,h),l}const Eo={kernelName:Je,backendName:`wasm`,setupFunc:wo,kernelFunc:To};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Do;function Oo(e){Do=e.wasm.cwrap(`Gather`,null,[`number`,`number`,`array`,`number`,`number`,`number`,`array`,`number`])}function ko(e){let{backend:t,inputs:n,attrs:r}=e,{x:i,indices:a}=n,{axis:o,batchDims:s}=r,c=qn(o,i.shape)[0],l=t.readSync(a.dataId),u=i.shape[c];for(let e=0;e<l.length;++e){let t=l[e];q(t<=u-1&&t>=0,()=>`GatherV2: the index value ${t} is not in [0, ${u-1}]`)}let d=Pt(i,a,c,s),f=Q({inputs:{x:i},attrs:{shape:[d.batchSize,d.outerSize,d.dimSize,d.sliceSize]},backend:t}),p=G(a.shape),m=Q({inputs:{x:a},attrs:{shape:[d.batchSize,p/d.batchSize]},backend:t}),h=[d.batchSize,d.outerSize,p/d.batchSize,d.sliceSize],g=t.makeOutput(h,i.dtype);if(G(i.shape)===0)return g;let _=f.shape.length-1,v=t.dataIdMap.get(f.dataId).id,y=t.dataIdMap.get(m.dataId).id,b=t.dataIdMap.get(g.dataId).id,x=new Uint8Array(new Int32Array(K(f.shape)).buffer),S=new Uint8Array(new Int32Array(K(h)).buffer);return Do(v,J[i.dtype],x,_,y,d.batchSize,S,b),t.disposeData(f.dataId),t.disposeData(m.dataId),g.shape=d.outputShape,g}const Ao={kernelName:jn,backendName:`wasm`,setupFunc:Oo,kernelFunc:ko},jo=X(Yn,!1,`bool`),Mo=X(rt,!1,`bool`),No=Y(De,`bool`),Po=Y(m,`bool`),Fo=Y(Rn,`bool`)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Io;function Lo(e){Io=e.wasm.cwrap(v,null,[`number`,`number`,`number`,`number`])}function Ro(e){let{inputs:{x:t},attrs:{alpha:n},backend:r}=e,i=r.dataIdMap.get(t.dataId).id,a=r.makeOutput(t.shape,`float32`);if(G(t.shape)!==0){let e=r.dataIdMap.get(a.dataId).id;Io(i,J[t.dtype],n,e)}return a}const zo={kernelName:v,backendName:`wasm`,setupFunc:Lo,kernelFunc:Ro},Bo=X(B,!1,`bool`),Vo=X(Ut,!1,`bool`)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2023 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ho;function Uo(e){Ho=e.wasm.cwrap(a,null,[`number`,`number`,`number`,`number`])}function Wo(e){let{attrs:t,backend:n}=e,{start:r,stop:i,num:a}=t,o=Math.floor(a),s=n.makeOutput([o],`float32`);return Ho(n.dataIdMap.get(s.dataId).id,r,i,o),s}const Go={kernelName:a,backendName:`wasm`,setupFunc:Uo,kernelFunc:Wo},Ko=Y(`Log`),qo=Y(se),Jo=X(z,!1,`bool`),Yo=Y(he),Xo=X(E,!1,`bool`),Zo=X(N,!1,`bool`)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Qo;function $o(e){Qo=e.wasm.cwrap(`LRN`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function es(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r;if(i.dtype!==`float32`)throw Error(`LRN error: x must have dtype float32`);let l=n.makeOutput(i.shape,i.dtype);return Qo(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(l.dataId).id,i.shape[3],a,o,s,c),l}const ts={kernelName:`LRN`,backendName:`wasm`,setupFunc:$o,kernelFunc:es};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ns;function rs(e){ns=e.wasm.cwrap(C,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function is(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r;if(i.dtype!==`float32`||a.dtype!==`float32`||o.dtype!==`float32`)throw Error(`LRNGrad error: x, y, and dy must have dtype float32`);let d=n.makeOutput(i.shape,i.dtype);return ns(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(o.dataId).id,n.dataIdMap.get(d.dataId).id,o.shape[3],s,c,l,u),d}const as={kernelName:C,backendName:`wasm`,setupFunc:rs,kernelFunc:is};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let os;function ss(e){os=e.wasm.cwrap(`Max`,null,[`number`,`number`,`number`,`number`])}function cs(e){let{backend:t,inputs:n,attrs:r}=e,{reductionIndices:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=o,{transposed:l,axes:u,originalAxes:d,inputWasTransposed:f}=xr(o,i,t);if(f){let e=t.dataIdMap.get(l.dataId).id;c=l,s=e}let p=c.shape.length;W(`max`,u,p);let[m,h]=Mn(c.shape,u),g=G(h),_=t.makeOutput(m,o.dtype);if(G(c.shape)!==0){let e=t.dataIdMap.get(_.dataId).id;os(s,J[o.dtype],g,e)}return f&&t.disposeData(l.dataId),a&&(_.shape=Xn(_.shape,d)),_}const ls={kernelName:`Max`,backendName:`wasm`,setupFunc:ss,kernelFunc:cs},us=X(Ne,!1)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ds;function fs(e){ds=e.wasm.cwrap(xe,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ps(e){let{inputs:t,attrs:n,backend:r}=e,i=t.x,a=r.dataIdMap.get(i.dataId).id;q(i.dtype===`float32`,()=>`Error in MaxPool: only float32 input is supported. Got ${i.dtype}.`);let{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=n,u=Hn(i.shape,o,s,1,c,l),d=u.filterHeight,f=u.filterWidth,p=u.padInfo.top,m=u.padInfo.right,h=u.padInfo.bottom,g=u.padInfo.left,_=u.dilationHeight,v=u.dilationWidth,y=u.strideHeight,b=u.strideWidth,x=u.inChannels,S=u.outChannels;if(u.dataFormat!==`channelsLast`)throw Error(`wasm backend does not support dataFormat:'${u.dataFormat}'. Please use 'channelsLast'.`);let C=r.makeOutput(u.outShape,`float32`),w=r.dataIdMap.get(C.dataId).id;return ds(a,i.shape[0],i.shape[1],i.shape[2],d,f,p,m,h,g,_,v,y,b,x,S,w),C}const ms={kernelName:xe,backendName:`wasm`,setupFunc:fs,kernelFunc:ps};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let hs;function gs(e){hs=e.wasm.cwrap(`MaxPool3D`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function _s(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=Ee(i.shape,a,o,1,s,c,l),d=n.makeOutput(u.outShape,i.dtype);return hs(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left),d}const vs={kernelName:$n,backendName:`wasm`,setupFunc:gs,kernelFunc:_s};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ys;function bs(e){ys=e.wasm.cwrap(`MaxPool3DGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function xs(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r,u=Ee(a.shape,o,s,1,c,l),d=n.makeOutput(a.shape,a.dtype);return ys(n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left),d}const Ss={kernelName:c,backendName:`wasm`,setupFunc:bs,kernelFunc:xs};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Cs;function ws(e){Cs=e.wasm.cwrap(`MaxPoolGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Ts(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r,u=Hn(a.shape,o,s,1,c,l),d=n.makeOutput(a.shape,a.dtype);return Cs(n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.top,u.padInfo.left),d}const Es={kernelName:Ie,backendName:`wasm`,setupFunc:ws,kernelFunc:Ts};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ds;function Os(e){Ds=e.wasm.cwrap(`MaxPoolWithArgmax`,null,[`number`,`number`,`number`,`number`,`boolean`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ks(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,includeBatchInIndex:c}=r;q(i.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.shape.length}.`);let l=[1,1];q(Ln(o,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '${l}'`);let u=Hn(i.shape,a,o,[1,1],s),d=n.makeOutput(u.outShape,i.dtype),f=n.makeOutput(u.outShape,`int32`);return Ds(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,n.dataIdMap.get(f.dataId).id,J[i.dtype],c,u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.top,u.padInfo.left),[d,f]}const As={kernelName:j,backendName:`wasm`,setupFunc:Os,kernelFunc:ks};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let js;function Ms(e){js=e.wasm.cwrap(ze,null,[`number, number, number`])}function Ns(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=xr(o,i,t),m=d;if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e,m=Wn(m.length,l.shape.length))}W(`mean`,m,l.shape.length);let[h,g]=Mn(l.shape,m),_=G(g),v=l;l.dtype!==`float32`&&(v=$({backend:t,inputs:{x:l},attrs:{dtype:`float32`}}),c=t.dataIdMap.get(v.dataId).id);let y=t.makeOutput(h,`float32`);if(G(l.shape)!==0){let e=t.dataIdMap.get(y.dataId).id;js(c,_,e)}return p&&t.disposeData(u.dataId),a&&(y.shape=Xn(y.shape,f)),l.dtype!==`float32`&&t.disposeData(v.dataId),y}const Ps={kernelName:ze,backendName:`wasm`,setupFunc:Ms,kernelFunc:Ns};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Fs;function Is(e){Fs=e.wasm.cwrap(`Min`,null,[`number`,`number`,`number`,`number`])}function Ls(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=xr(o,i,t);if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e)}let m=l.shape.length;W(`min`,d,m);let[h,g]=Mn(l.shape,d),_=G(g),v=t.makeOutput(h,l.dtype);if(G(l.shape)!==0){let e=t.dataIdMap.get(v.dataId).id;Fs(c,J[o.dtype],_,e)}return p&&t.disposeData(u.dataId),a&&(v.shape=Xn(v.shape,f)),v}const Rs={kernelName:`Min`,backendName:`wasm`,setupFunc:Is,kernelFunc:Ls},zs=X(ie,!1)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Bs;(function(e){e[e.reflect=0]=`reflect`,e[e.symmetric=1]=`symmetric`})(Bs||={});let Vs;function Hs(e){Vs=e.wasm.cwrap(dn,null,[`number`,`array`,`number`,`number`,`array`,`array`,`number`,`number`])}function Us(e){let{inputs:{x:t},backend:n,attrs:{paddings:r,mode:i}}=e,a=r.map((e,n)=>e[0]+t.shape[n]+e[1]),o=n.dataIdMap.get(t.dataId).id,s=n.makeOutput(a,t.dtype),c=n.dataIdMap.get(s.dataId).id,l=new Uint8Array(new Int32Array(t.shape).buffer),u=r.map(e=>e[0]),d=r.map(e=>e[1]),f=new Uint8Array(new Int32Array(u).buffer),p=new Uint8Array(new Int32Array(d).buffer);return Vs(o,l,t.shape.length,J[t.dtype],f,p,Bs[i],c),s}const Ws={kernelName:dn,backendName:`wasm`,kernelFunc:Us,setupFunc:Hs};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Gs;function Ks(e){Gs=e.wasm.cwrap(ce,null,[`number`,`number`,`number`,`number`])}function qs(e){let{backend:t,inputs:{logits:n},attrs:{dim:r}}=e,i=t.dataIdMap.get(n.dataId).id,a=t.makeOutput(n.shape,n.dtype),o=t.dataIdMap.get(a.dataId).id,s=n.shape[r],c=G(n.shape)/s;return G(a.shape)===0||Gs(i,o,s,c),a}const Js={kernelName:ce,backendName:`wasm`,setupFunc:Ks,kernelFunc:qs};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ys;function Xs(e){Ys=e.wasm.cwrap(We,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function Zs(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r;if(i.dtype!==`float32`)throw Error(`Tensor logits must have dtype float32, got ${i.dtype}`);let c=s?i:qs({inputs:{logits:i},backend:n,attrs:{dim:i.shape.length-1}}),[l,u]=c.shape,d=n.makeOutput([l,a],`int32`);return Ys(n.dataIdMap.get(c.dataId).id,l,u,a,o,n.dataIdMap.get(d.dataId).id),s||n.disposeData(c.dataId),d}const Qs={kernelName:We,backendName:`wasm`,setupFunc:Xs,kernelFunc:Zs},$s=X(`Mod`,!0),ec=X(U,!0),tc=Y(`Neg`)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nc(e,t){let n=new Int32Array(e.wasm.HEAPU8.buffer,t,4),r=n[0],i=n[1],a=n[2],o=n[3];return e.wasm._free(t),{pSelectedIndices:r,selectedSize:i,pSelectedScores:a,pValidOutputs:o}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let rc;function ic(e){rc=e.wasm.cwrap(ye,`number`,[`number`,`number`,`number`,`number`,`number`])}function ac(e){let{backend:t,inputs:n,attrs:r}=e,{iouThreshold:i,maxOutputSize:a,scoreThreshold:o}=r,{boxes:s,scores:c}=n,l=t.dataIdMap.get(s.dataId).id,u=t.dataIdMap.get(c.dataId).id,{pSelectedIndices:d,selectedSize:f,pSelectedScores:p,pValidOutputs:m}=nc(t,rc(l,u,a,i,o));return t.wasm._free(p),t.wasm._free(m),t.makeOutput([f],`int32`,d)}const oc={kernelName:ye,backendName:`wasm`,setupFunc:ic,kernelFunc:ac};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let sc;function cc(e){sc=e.wasm.cwrap(t,`number`,[`number`,`number`,`number`,`number`,`number`,`bool`])}function lc(e){let{backend:t,inputs:n,attrs:r}=e,{iouThreshold:i,maxOutputSize:a,scoreThreshold:o,padToMaxOutputSize:s}=r,{boxes:c,scores:l}=n,u=t.dataIdMap.get(c.dataId).id,d=t.dataIdMap.get(l.dataId).id,{pSelectedIndices:f,selectedSize:p,pSelectedScores:m,pValidOutputs:h}=nc(t,sc(u,d,a,i,o,s));return t.wasm._free(m),[t.makeOutput([p],`int32`,f),t.makeOutput([],`int32`,h)]}const uc={kernelName:t,backendName:`wasm`,setupFunc:cc,kernelFunc:lc};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let dc;function fc(e){dc=e.wasm.cwrap(yt,`number`,[`number`,`number`,`number`,`number`,`number`,`number`])}function pc(e){let{backend:t,inputs:n,attrs:r}=e,{iouThreshold:i,maxOutputSize:a,scoreThreshold:o,softNmsSigma:s}=r,{boxes:c,scores:l}=n,u=t.dataIdMap.get(c.dataId).id,d=t.dataIdMap.get(l.dataId).id,{pSelectedIndices:f,selectedSize:p,pSelectedScores:m,pValidOutputs:h}=nc(t,dc(u,d,a,i,o,s));return t.wasm._free(h),[t.makeOutput([p],`int32`,f),t.makeOutput([p],`float32`,m)]}const mc={kernelName:yt,backendName:`wasm`,setupFunc:fc,kernelFunc:pc},hc=X(Cn,!1,`bool`)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let gc;function _c(e){gc=e.wasm.cwrap(rn,null,[`number`,`number`,`number`,`number`,`number`])}function vc(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r,l=n.makeOutput([...i.shape,o],a),u=n.dataIdMap.get(l.dataId).id,d=n.dataIdMap.get(i.dataId).id;return gc(d,o,s,c,u),l}const yc={kernelName:rn,backendName:`wasm`,setupFunc:_c,kernelFunc:vc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function bc(e){let{inputs:{x:t},backend:n}=e,r=n.makeOutput(t.shape,t.dtype);return n.typedArrayFromHeap(r).fill(1),r}const xc={kernelName:hn,backendName:`wasm`,kernelFunc:bc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Sc(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return $a({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{pt(a,e.shape,`All tensors passed to stack must have matching shapes`),q(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=Pi({inputs:t.map(e=>{let t=$a({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeData(e.dataId)),c}const Cc={kernelName:Lt,backendName:`wasm`,kernelFunc:Sc};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let wc;function Tc(e){wc=e.wasm.cwrap($e,null,[`number`,`array`,`number`,`number`,`array`,`array`,`number`,`number`])}function Ec(e){let{inputs:{x:t},backend:n,attrs:{paddings:r,constantValue:i}}=e,a=r.map((e,n)=>e[0]+t.shape[n]+e[1]);if(G(t.shape)===0)return no({backend:n,attrs:{shape:a,value:i,dtype:t.dtype}});let o=n.dataIdMap.get(t.dataId).id,s=n.makeOutput(a,t.dtype),c=n.dataIdMap.get(s.dataId).id,l=new Uint8Array(new Int32Array(t.shape).buffer),u=r.map(e=>e[0]),d=r.map(e=>e[1]),f=new Uint8Array(new Int32Array(u).buffer),p=new Uint8Array(new Int32Array(d).buffer);return wc(o,l,t.shape.length,J[t.dtype],f,p,i,c),s}const Dc={kernelName:$e,backendName:`wasm`,kernelFunc:Ec,setupFunc:Tc},Oc=X(`Pow`,!1)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let kc;function Ac(e){kc=e.wasm.cwrap(yn,null,[`number`,`number`,`number`])}function jc(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t,a=n.dataIdMap.get(r.dataId).id,o=n.dataIdMap.get(i.dataId).id,s=a,c=r,l=c;c.dtype!==`float32`&&(l=$({backend:n,inputs:{x:r},attrs:{dtype:`float32`}}),s=n.dataIdMap.get(l.dataId).id);let u=n.makeOutput(r.shape,`float32`),d=n.dataIdMap.get(u.dataId).id;return kc(s,o,d),c.dtype!==`float32`&&n.disposeData(l.dataId),u}const Mc={kernelName:yn,backendName:`wasm`,setupFunc:Ac,kernelFunc:jc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Nc;function Pc(e){Nc=e.wasm.cwrap(ut,null,[`number`,`number`,`number`,`number`])}function Fc(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=xr(o,i,t),m=d;if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e,m=Wn(m.length,l.shape.length))}W(`prod`,m,l.shape.length);let[h,g]=Mn(l.shape,m),_=G(g),v=t.makeOutput(h,l.dtype);if(G(l.shape)!==0){let e=t.dataIdMap.get(v.dataId).id;Nc(c,_,J[v.dtype],e)}return p&&t.disposeData(u.dataId),a&&(v.shape=Xn(v.shape,f)),v}const Ic={kernelName:ut,backendName:`wasm`,setupFunc:Pc,kernelFunc:Fc},Lc={kernelName:qt,backendName:`wasm`,kernelFunc:e=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{backend:t,attrs:n}=e,{start:r,stop:i,step:a,dtype:o}=n,s=oi(r,i,a,o),c=t.makeOutput([s.length],o);return t.typedArrayFromHeap(c).set(s),c}},Rc=X(Dn,!0),zc=Y(ht),Bc=Y(wt),Vc=Y(ln)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Hc;function Uc(e){Hc=e.wasm.cwrap(Nt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Wc(e){let{backend:t,inputs:n,attrs:r}=e,{images:i}=n,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,[u,d,f,p]=i.shape,m=[u,c,l,p],h=t.dataIdMap.get(i.dataId),g;h.dtype!==`float32`&&(g=$({backend:t,inputs:{x:i},attrs:{dtype:`float32`}}),h=t.dataIdMap.get(g.dataId));let _=h.id,v=t.makeOutput(m,`float32`);if(G(i.shape)===0)return v;let y=t.dataIdMap.get(v.dataId).id;return Hc(_,u,d,f,p,c,l,+!!a,+!!o,y),g!=null&&t.disposeData(g.dataId),v}const Gc={kernelName:Nt,backendName:`wasm`,setupFunc:Uc,kernelFunc:Wc};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Kc;function qc(e){Kc=e.wasm.cwrap(Ot,null,[`number`,`number`,`number`,`array`,`array`,`boolean`])}function Jc(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=n.makeOutput(i.shape,`float32`),c=n.dataIdMap.get(i.dataId),l;return c.dtype!==`float32`&&(l=$({backend:n,inputs:{x:i},attrs:{dtype:`float32`}}),c=n.dataIdMap.get(l.dataId)),Kc(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(s.dataId).id,new Uint8Array(new Int32Array(i.shape).buffer),new Uint8Array(new Int32Array(a.shape).buffer),o),l!=null&&n.disposeData(l.dataId),s}const Yc={kernelName:Ot,backendName:`wasm`,setupFunc:qc,kernelFunc:Jc};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the 'License');
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an 'AS IS' BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Xc;function Zc(e){Xc=e.wasm.cwrap(Ye,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Qc(e){let{backend:t,inputs:n,attrs:r}=e,{images:i}=n,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,[u,d,f,p]=i.shape,m=[u,c,l,p],h=t.makeOutput(m,`float32`);if(G(i.shape)===0)return h;let g=t.dataIdMap.get(i.dataId),_;g.dtype!==`float32`&&(_=$({backend:t,inputs:{x:i},attrs:{dtype:`float32`}}),g=t.dataIdMap.get(_.dataId));let v=g.id,y=t.dataIdMap.get(h.dataId).id;return Xc(v,u,d,f,p,c,l,+!!a,+!!o,y),_!=null&&t.disposeData(_.dataId),h}const $c={kernelName:Ye,backendName:`wasm`,setupFunc:Zc,kernelFunc:Qc};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let el;function tl(e){el=e.wasm.cwrap(Nn,null,[`number`,`number`,`number`,`array`,`array`,`boolean`])}function nl(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=n.makeOutput(i.shape,`float32`),c=n.dataIdMap.get(i.dataId),l;return c.dtype!==`float32`&&(l=$({backend:n,inputs:{x:i},attrs:{dtype:`float32`}}),c=n.dataIdMap.get(l.dataId)),el(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(s.dataId).id,new Uint8Array(new Int32Array(i.shape).buffer),new Uint8Array(new Int32Array(a.shape).buffer),o),l!=null&&n.disposeData(l.dataId),s}const rl={kernelName:Nn,backendName:`wasm`,setupFunc:tl,kernelFunc:nl};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let il;function al(e){il=e.wasm.cwrap(Zn,null,[`number`,`array`,`number`,`array`,`number`,`number`])}function ol(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r,o=qn(a,i.shape);if(i.shape.length===0)return mr({inputs:{x:i},backend:n});let s=n.makeOutput(i.shape,i.dtype),c=n.dataIdMap.get(i.dataId).id,l=n.dataIdMap.get(s.dataId).id,u=new Uint8Array(new Int32Array(o).buffer),d=new Uint8Array(new Int32Array(i.shape).buffer);il(c,u,o.length,d,i.shape.length,l);let f=Q({inputs:{x:s},attrs:{shape:i.shape},backend:n});return n.disposeData(s.dataId),f}const sl={kernelName:Zn,backendName:`wasm`,kernelFunc:ol,setupFunc:al};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let cl;function ll(e){cl=e.wasm.cwrap(at,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`])}function ul(e){let{inputs:t,backend:n,attrs:r}=e,{image:i}=t,{radians:a,fillValue:o,center:s}=r,c=n.makeOutput(i.shape,i.dtype),l=n.dataIdMap.get(i.dataId).id,d=n.dataIdMap.get(c.dataId).id,[f,p,m,h]=i.shape,[g,_]=u(s,p,m),v=typeof o==`number`?[o,o,o,o===0?0:255]:[...o,255],y=new Uint8Array(new Int32Array(v).buffer);return cl(l,f,p,m,h,a,g,_,y,v.length,d),c}const dl={kernelName:at,backendName:`wasm`,kernelFunc:ul,setupFunc:ll},fl=Y(Gn),pl=Y(H)
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ml;function hl(e){ml=e.wasm.cwrap(h,null,[`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`])}function gl(e){let{backend:t,inputs:n,attrs:r}=e,{indices:i,updates:a}=n,{shape:o}=r,s=t.makeOutput(o,a.dtype);if(G(o)===0)return s;let{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=bn(a,i,o),p=t.dataIdMap.get(i.dataId).id,m=t.dataIdMap.get(a.dataId).id,h=new Uint8Array(new Int32Array(d).buffer),g=t.dataIdMap.get(s.dataId).id;return ml(p,m,J[a.dtype],c,l,u,h,f,g),s}const _l={kernelName:h,backendName:`wasm`,setupFunc:hl,kernelFunc:gl};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let vl;function yl(e){vl=e.wasm.cwrap(zn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`bool`,`number`])}function bl(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r;if(i.dtype!==a.dtype)throw Error(`SearchSorted error: sorted_sequence must have the same dtype as values. Got ${i.dtype} and ${a.dtype}`);let s=n.makeOutput(a.shape,`int32`);function c(e){return n.dataIdMap.get(e.dataId).id}return vl(c(i),c(a),i.shape[0],i.shape[1],a.shape[1],J[i.dtype],o===`left`,c(s)),s}const xl={kernelName:zn,backendName:`wasm`,setupFunc:yl,kernelFunc:bl};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Sl;function Cl(e){Sl=e.wasm.cwrap(`SelectV2`,null,[`number`,`number`,`number`,`number`,`number`])}function wl(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t,o=n.dataIdMap.get(r.dataId).id,s=n.dataIdMap.get(i.dataId).id,c=n.dataIdMap.get(a.dataId).id,l=n.makeOutput(i.shape,i.dtype),u=n.dataIdMap.get(l.dataId).id,d=r.shape.length,f=i.shape.length,p=d===0||d>1||f===1?1:G(i.shape.slice(1));return Sl(o,s,c,p,u),l}const Tl={kernelName:je,backendName:`wasm`,kernelFunc:wl,setupFunc:Cl},El=Y(w)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Dl;function Ol(e){Dl=e.wasm.cwrap(y,null,[`number`,`number`])}function kl(e){let{backend:t,inputs:{x:n}}=e,r=t.dataIdMap.get(n.dataId).id,i=t.makeOutput(n.shape,n.dtype),a=t.dataIdMap.get(i.dataId).id;return G(i.shape)===0||Dl(r,a),i}const Al={kernelName:`Sigmoid`,backendName:`wasm`,setupFunc:Ol,kernelFunc:kl},jl=Y(pe),Ml=Y(`Sin`),Nl=Y(o),Pl=Y(ue)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Fl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r,s=G(a),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=Dc.kernelFunc({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),u=Kn(l.shape,a,s,!1),d=On(u.length,a.length,!1),f=et(l.shape,a,s,!1),p=Q({inputs:{x:l},backend:n,attrs:{shape:u}}),m=Z({inputs:{x:p},backend:n,attrs:{perm:d}}),h=Q({inputs:{x:m},backend:n,attrs:{shape:f}});return n.disposeData(l.dataId),n.disposeData(p.dataId),n.disposeData(m.dataId),h}const Il={kernelName:ge,backendName:`wasm`,kernelFunc:Fl};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ll;function Rl(e){Ll=e.wasm.cwrap(`SparseFillEmptyRows`,`number`,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function zl(e){let{backend:t,inputs:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=n,s=r.shape[0],c=r.shape[1],l=t.readSync(a.dataId)[0],u=[s+l,c],d=t.dataIdMap.get(r.dataId).id,f=t.dataIdMap.get(i.dataId).id,p=t.dataIdMap.get(o.dataId).id,m=t.makeOutput(u,r.dtype),h=t.dataIdMap.get(m.dataId).id,g=t.makeOutput(u.slice(0,1),i.dtype),_=t.dataIdMap.get(g.dataId).id,v=t.makeOutput([l],`bool`),y=t.dataIdMap.get(v.dataId).id,b=t.makeOutput([s],r.dtype),x=t.dataIdMap.get(b.dataId).id,S=t.makeOutput([4],`int32`),C=t.dataIdMap.get(S.dataId).id,w=Ll(d,f,J[i.dtype],s,l,c,p,h,_,y,x,C),T=t.readSync(S.dataId),E;switch(T[0]){case 1:E=Jt(T[1]);break;case 2:E=kt(T[1],T[2]);break;case 3:E=Tt(T[1],T[2],T[3]);break;default:E=``}if(t.disposeData(S.dataId),E)throw t.disposeData(m.dataId),t.disposeData(g.dataId),t.disposeData(v.dataId),t.disposeData(b.dataId),Error(E);let D=m,O=g;return w!==u[0]&&(D=mi({inputs:{x:m},attrs:{begin:0,size:[w,c]},backend:t}),O=mi({inputs:{x:g},attrs:{begin:0,size:w},backend:t}),t.disposeData(m.dataId),t.disposeData(g.dataId)),[D,O,v,b]}const Bl={kernelName:D,backendName:`wasm`,setupFunc:Rl,kernelFunc:zl};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Vl;function Hl(e){Vl=e.wasm.cwrap(P,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Ul(e){let{backend:t,inputs:n}=e,{inputIndices:r,inputShape:i,newShape:a}=n;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape
        ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=t.dataIdMap.get(r.dataId).id,s=t.dataIdMap.get(i.dataId).id,c=t.dataIdMap.get(a.dataId).id,l=r.shape[0],u=G(a.shape),d=t.makeOutput([l,u],r.dtype),f=t.dataIdMap.get(d.dataId).id,p=t.makeOutput([u],a.dtype),m=t.dataIdMap.get(p.dataId).id,h=t.makeOutput([3],`int32`),g=t.dataIdMap.get(h.dataId).id;Vl(o,s,c,l,f,m,g);let _=t.readSync(h.dataId),v;switch(_[0]){case 0:v=bt(_[1],_[2]);break;case 1:v=sn(_[1],_[2]);break;case 2:v=Wt();break;case 3:{let e=Array.from(t.readSync(i.dataId)),n=Array.from(t.readSync(p.dataId));v=dt(e,n);break}case 4:{let e=Array.from(t.readSync(i.dataId)),n=Array.from(t.readSync(p.dataId));v=wn(e,n);break}default:v=``}if(t.disposeData(h.dataId),v)throw t.disposeData(d.dataId),t.disposeData(p.dataId),Error(v);return[d,p]}const Wl={kernelName:P,backendName:`wasm`,setupFunc:Hl,kernelFunc:Ul};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Gl;function Kl(e){Gl=e.wasm.cwrap(`SparseSegmentReduction`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ql(e,t){let{backend:n,inputs:r}=e,{data:i,indices:a,segmentIds:o}=r,s=a.shape[0],c=n.readSync(o.dataId,s-1,s)[0],l=s>0?c+1:0;if(l<0)throw Error(an());let u=i.shape.slice();u[0]=l;let d=n.dataIdMap.get(i.dataId).id,f=n.dataIdMap.get(a.dataId).id,p=n.dataIdMap.get(o.dataId).id,m=n.makeOutput(u,i.dtype),h=n.dataIdMap.get(m.dataId).id,g=n.makeOutput([4],`int32`),_=n.dataIdMap.get(g.dataId).id;Gl(d,J[i.dtype],i.shape[0],f,p,h,_,t,0);let v=n.readSync(g.dataId),y;switch(v[0]){case 0:y=an();break;case 1:y=gn();break;case 2:y=ot(v[1],v[2]);break;case 3:y=Xe(v[1],v[2],v[3]);break;default:y=``}if(n.disposeData(g.dataId),y)throw n.disposeData(m.dataId),Error(y);return m}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Jl(e){return ql(e,!0)}const Yl={kernelName:I,backendName:`wasm`,setupFunc:Kl,kernelFunc:Jl};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Xl(e){return ql(e,!1)}const Zl={kernelName:Se,backendName:`wasm`,setupFunc:Kl,kernelFunc:Xl};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ql;function $l(e){Ql=e.wasm.cwrap(er,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`])}function eu(e){let{backend:t,inputs:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=n,{outputShape:s}=r,c=t.makeOutput(s,o.dtype);if(G(s)===0)return c;let{sliceRank:l,numUpdates:u,sliceSize:d,strides:f,outputSize:p}=bn(a,i,s),m=t.dataIdMap.get(i.dataId).id,h=t.dataIdMap.get(a.dataId).id,g=t.dataIdMap.get(o.dataId).id,_=new Uint8Array(new Int32Array(f).buffer),v=t.dataIdMap.get(c.dataId).id;return Ql(m,h,a.shape.length,g,J[o.dtype],l,u,d,_,p,v),c}const tu={kernelName:er,backendName:`wasm`,setupFunc:$l,kernelFunc:eu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nu(e){let{inputs:t,attrs:n,backend:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=n,s=qn(o,i.shape)[0],c=Ke(i,a,s),l=Array(i.shape.length).fill(0),u=i.shape.slice();return c.map(e=>{let t=[...u];t[s]=e;let n=mi({inputs:{x:i},attrs:{begin:l,size:t},backend:r});return l[s]+=e,n})}const ru={kernelName:l,backendName:`wasm`,kernelFunc:nu},iu=Y(Le),au=Y(M),ou=X(Pe,!0)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let su;function cu(e){su=e.wasm.cwrap(Be,null,[`number`,`number`,`number`,`number`])}function lu(e){let{backend:t,inputs:n,attrs:r}=e,{alpha:i}=r,{x:a}=n,o=t.dataIdMap.get(a.dataId).id,s=t.makeOutput(a.shape,a.dtype),c=t.dataIdMap.get(s.dataId).id;return su(o,i,J[a.dtype],c),s}const uu={kernelName:Be,backendName:`wasm`,setupFunc:cu,kernelFunc:lu};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let du;function fu(e){du=e.wasm.cwrap(k,null,[`number`,`array`,`number`,`array`,`array`,`array`,`array`,`array`,`number`,`number`])}function pu(e){let{backend:t,inputs:n,attrs:i}=e,{x:a}=n,{begin:o,end:s,strides:c,beginMask:l,endMask:u,ellipsisMask:d,newAxisMask:f,shrinkAxisMask:p}=i,{finalShapeSparse:m,finalShape:h,isIdentity:_,sliceDim0:v,isSimpleSlice:y,begin:b,end:x,strides:S}=r(a.shape,o,s,c,l,u,d,f,p),C;if(_)C=Q({inputs:{x:a},backend:t,attrs:{shape:h}});else if(v||y){q(a.shape.length>=1,()=>`Input must have rank at least 1, got: ${a.shape.length}`);let e=g(b,x,S),n=mi({inputs:{x:a},backend:t,attrs:{begin:b,size:e}});C=Q({inputs:{x:n},backend:t,attrs:{shape:h}}),t.disposeData(n.dataId)}else{let e=t.makeOutput(m,`float32`),n=t.dataIdMap.get(a.dataId).id,r=new Uint8Array(new Int32Array(K(a.shape)).buffer),i=new Uint8Array(new Int32Array(b).buffer),o=new Uint8Array(new Int32Array(x).buffer),s=new Uint8Array(new Int32Array(S).buffer),c=new Uint8Array(new Int32Array(m).buffer),l=new Uint8Array(new Int32Array(K(m)).buffer),u=t.dataIdMap.get(e.dataId).id;du(n,r,a.shape.length,i,o,s,c,l,m.length,u),C=Q({inputs:{x:e},backend:t,attrs:{shape:h}}),t.disposeData(e.dataId)}return C}const mu={kernelName:k,backendName:`wasm`,setupFunc:fu,kernelFunc:pu};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function hu(e){let{backend:t,inputs:n,attrs:r}=e,{data:i,dataSplits:a}=n,{separator:o,nGramWidths:s,leftPad:c,rightPad:l,padWidth:u,preserveShortSequences:d}=r,[f,p]=li(t.readSync(i.dataId),t.readSync(a.dataId),o,s,c,l,u,d),m=t.makeOutput([f.length],`string`),h=t.dataIdMap.get(m.dataId);h.stringBytes=f;let g=t.makeOutput(a.shape,`int32`);return t.typedArrayFromHeap(g).set(p),[m,g]}const gu={kernelName:ae,backendName:`wasm`,kernelFunc:hu};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function _u(e){let{backend:t,inputs:n,attrs:r}=e,{input:i,delimiter:a}=n,{skipEmpty:o}=r,[s,c,l]=di(t.readSync(i.dataId),t.readSync(a.dataId)[0],o),u=c.length,d=t.makeOutput([u,2],`int32`);t.typedArrayFromHeap(d).set(s);let f=t.makeOutput([u],`string`),p=t.dataIdMap.get(f.dataId);p.stringBytes=c;let m=t.makeOutput([2],`int32`);return t.typedArrayFromHeap(m).set(l),[d,f,m]}const vu={kernelName:fn,backendName:`wasm`,kernelFunc:_u};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function yu(e){let{backend:t,inputs:n,attrs:r}=e,{input:i}=n,{numBuckets:a}=r,o=fi(t.readSync(i.dataId),a),s=t.makeOutput(i.shape,`int32`);return t.typedArrayFromHeap(s).set(o),s}const bu={kernelName:te,backendName:`wasm`,kernelFunc:yu},xu=X(`Sub`,!0)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Su;function Cu(e){Su=e.wasm.cwrap(`Sum`,null,[`number`,`number`,`number`,`number`])}function wu(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=xr(o,i,t),m=d;if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e,m=Wn(m.length,l.shape.length))}W(`sum`,m,l.shape.length);let[h,g]=Mn(l.shape,m),_=G(g),v=t.makeOutput(h,l.dtype);if(G(l.shape)!==0){let e=t.dataIdMap.get(v.dataId).id;Su(c,_,J[v.dtype],e)}return p&&t.disposeData(u.dataId),a&&(v.shape=Xn(v.shape,f)),v}const Tu={kernelName:`Sum`,backendName:`wasm`,setupFunc:Cu,kernelFunc:wu},Eu=Y(`Tan`),Du=Y(be)
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ou;function ku(e){Ou=e.wasm.cwrap(n,null,[`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`,`number`])}function Au(e){let{backend:t,inputs:n,attrs:r}=e,{tensor:i,indices:a,updates:o}=n,{}=r,s=t.makeOutput(i.shape,i.dtype);if(G(i.shape)===0)return s;let{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=bn(o,a,i.shape),p=t.dataIdMap.get(a.dataId).id,m=t.dataIdMap.get(o.dataId).id,h=t.dataIdMap.get(i.dataId).id,g=new Uint8Array(new Int32Array(d).buffer),_=t.dataIdMap.get(s.dataId).id;return Ou(p,m,J[o.dtype],c,l,u,g,f,_,h),s}const ju={kernelName:n,backendName:`wasm`,setupFunc:ku,kernelFunc:Au};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Mu;function Nu(e){Mu=e.wasm.cwrap(_t,null,[`number`,`array`,`number`,`array`,`number`,`number`])}function Pu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,a=n.dataIdMap.get(i.dataId).id,{reps:o}=r,s=Array(i.shape.length);for(let e=0;e<s.length;e++)s[e]=i.shape[e]*o[e];let c=new Uint8Array(new Int32Array(i.shape).buffer),l=new Uint8Array(new Int32Array(s).buffer),u=n.makeOutput(s,i.dtype),d=n.dataIdMap.get(u.dataId).id;return Mu(a,c,i.shape.length,l,s.length,J[u.dtype],d),u}const Fu={kernelName:_t,backendName:`wasm`,setupFunc:Nu,kernelFunc:Pu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Iu;function Lu(e){Iu=e.wasm.cwrap(xn,null,[`number`,`array`,`number`,`number`,`number`,`bool`,`number`,`number`])}const Ru={kernelName:xn,backendName:`wasm`,setupFunc:Lu,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r}=e,{k:i,sorted:a}=n,o=t.dataIdMap.get(r.dataId).id,s=new Uint8Array(new Int32Array(r.shape).buffer),c=r.shape.slice();c[c.length-1]=i;let l=t.makeOutput(c,r.dtype),u=t.dataIdMap.get(l.dataId).id,d=t.makeOutput(c,`int32`),f=t.dataIdMap.get(d.dataId).id;return Iu(o,s,r.shape.length,J[r.dtype],i,a,u,f),[l,d]}};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let zu;function Bu(e){zu=e.wasm.cwrap(tn,null,[`number`,`number`,`bool`,`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`array`,`number`,`number`,`number`,`number`,`number`])}function Vu(e){let{backend:t,inputs:n,attrs:r}=e,{image:i,transforms:a}=n,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=r,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=[u,m,h,p],_=new Uint8Array(new Int32Array(K(i.shape)).buffer),v=new Uint8Array(new Int32Array(K(g)).buffer),y=t.makeOutput(g,i.dtype),b=t.dataIdMap.get(y.dataId).id,x=t.dataIdMap.get(i.dataId).id,S=t.dataIdMap.get(a.dataId).id,C=o===`nearest`?1:2,w;switch(s){case`constant`:w=1;break;case`reflect`:w=2;break;case`wrap`:w=3;break;case`nearest`:w=4;break;default:w=1}return zu(x,S,a.shape[0]>1,u,m,h,p,f,d,_,i.shape.length-1,v,g.length-1,C,w,c,b),y}const Hu={kernelName:tn,backendName:`wasm`,setupFunc:Bu,kernelFunc:Vu};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Uu(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t,{outputValues:o,outputShape:s,indices:c}=pi(r.readSync(a.dataId),i,a.shape,a.dtype);return[r.makeOutput(s,a.dtype,void 0,o),r.makeOutput([c.length],`int32`,void 0,c)]}const Wu={kernelName:Ft,backendName:`wasm`,kernelFunc:Uu};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Gu(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i.shape[a],s=i.shape.length,c=Array(s-1),l=0;for(let e=0;e<s;e++)e!==a&&(c[l++]=i.shape[e]);let u=Array(o),d=Array(s).fill(0),f=i.shape.slice();f[a]=1;for(let e=0;e<u.length;e++)d[a]=e,u[e]=mi({inputs:{x:i},attrs:{begin:d,size:f},backend:n});return u.map(({dataId:e,dtype:t})=>({dataId:e,dtype:t,shape:c}))}const Ku={kernelName:Ze,backendName:`wasm`,kernelFunc:Gu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function qu(e){let{inputs:{x:t},backend:n}=e,r=n.makeOutput(t.shape,t.dtype);return n.typedArrayFromHeap(r).fill(0),r}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
const Ju=[ar,or,sr,cr,lr,pr,Tr,kr,jr,Mr,Nr,Pr,Fr,Ir,Lr,Vr,$r,Gr,Yr,ii,bi,wi,Ti,Di,Oi,ki,Ni,Fi,zi,Ui,qi,Zi,ta,na,ra,ca,fa,ga,ba,wa,Oa,Ma,Ia,Ba,Wa,Ga,Ya,Xa,Za,Qa,eo,to,ro,so,co,lo,mo,vo,So,Eo,Ao,jo,Mo,hr,No,Po,Fo,zo,Bo,Vo,Go,qo,Ko,Jo,Yo,Xo,Zo,ts,as,ls,us,ms,vs,Ss,Es,As,Ps,Rs,zs,Ws,Qs,$s,ec,tc,oc,uc,mc,hc,yc,xc,Cc,Dc,Oc,Mc,Ic,Lc,Rc,zc,Bc,Vc,ei,Gc,Yc,$c,rl,sl,dl,fl,pl,_l,xl,Tl,El,Al,jl,Ml,Nl,vi,Js,Pl,Il,Bl,Wl,Yl,Zl,tu,ru,iu,au,ou,uu,mu,gu,vu,bu,xu,Tu,Eu,Du,ju,Fu,Ru,Hu,br,Wu,Ku,{kernelName:on,backendName:`wasm`,kernelFunc:qu}];for(let e of Ju)xt(e);
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
const Yu=ct();Yu.registerFlag(`WASM_HAS_SIMD_SUPPORT`,async()=>{try{return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]))}catch{return!1}}),Yu.registerFlag(`WASM_HAS_MULTITHREAD_SUPPORT`,async()=>{if(Yu.get(`IS_NODE`))return!1;try{return new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)),WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]))}catch{return!1}});var Xu=/* @__PURE__ */ In(((e,t)=>{var n=(()=>{var e=typeof document<`u`&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<`u`&&(e||=__filename),(function(t){t||={};function n(){return M.buffer!=I&&z(M.buffer),L}function r(){return M.buffer!=I&&z(M.buffer),se}function i(){return M.buffer!=I&&z(M.buffer),R}function a(){return M.buffer!=I&&z(M.buffer),ce}function o(){return M.buffer!=I&&z(M.buffer),le}var s=t===void 0?{}:t,c,l;s.ready=new Promise(function(e,t){c=e,l=t});var u;typeof process<`u`&&process.listeners&&(u={uncaughtException:process.listeners(`uncaughtException`),unhandledRejection:process.listeners(`unhandledRejection`)});var d=Object.assign({},s),f=[],p=(e,t)=>{throw t},m=typeof window==`object`,h=typeof importScripts==`function`,g=typeof process==`object`&&typeof process.versions==`object`&&typeof process.versions.node==`string`,_=s.ENVIRONMENT_IS_PTHREAD||!1,v=``;function y(e){return s.locateFile?s.locateFile(e,v):v+e}var b,x,S;function C(e){e instanceof Ne||k(`exiting due to exception: `+e)}if(g){var w=re(),T=re();v=h?T.dirname(v)+`/`:__dirname+`/`,b=(e,t)=>(e=Oe(e)?new URL(e):T.normalize(e),w.readFileSync(e,t?void 0:`utf8`)),S=e=>{var t=b(e,!0);return t.buffer||(t=new Uint8Array(t)),t},x=(e,t,n)=>{e=Oe(e)?new URL(e):T.normalize(e),w.readFile(e,function(e,r){e?n(e):t(r.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,`/`),f=process.argv.slice(2),process.on(`uncaughtException`,function(e){if(!(e instanceof Ne))throw e}),process.on(`unhandledRejection`,function(e){throw e}),p=(e,t)=>{if(me())throw process.exitCode=e,t;C(t),process.exit(e)},s.inspect=function(){return`[Emscripten Module object]`};let e;try{e=re()}catch(e){throw console.error(`The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?`),e}global.Worker=e.Worker}else(m||h)&&(h?v=self.location.href:typeof document<`u`&&document.currentScript&&(v=document.currentScript.src),e!==void 0&&e&&(v=e),v=v.indexOf(`blob:`)===0?``:v.substr(0,v.replace(/[?#].*/,``).lastIndexOf(`/`)+1),g||(b=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.send(null),t.responseText},h&&(S=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.responseType=`arraybuffer`,t.send(null),new Uint8Array(t.response)}),x=(e,t,n)=>{var r=new XMLHttpRequest;r.open(`GET`,e,!0),r.responseType=`arraybuffer`,r.onload=()=>{if(r.status==200||r.status==0&&r.response){t(r.response);return}n()},r.onerror=n,r.send(null)}));g&&typeof performance>`u`&&(global.performance=re().performance);var E=console.log.bind(console),D=console.warn.bind(console);g&&(E=e=>w.writeSync(1,e+`
`),D=e=>w.writeSync(2,e+`
`));var O=s.print||E,k=s.printErr||D;Object.assign(s,d),d=null,s.arguments&&(f=s.arguments),s.thisProgram&&s.thisProgram,s.quit&&(p=s.quit),Atomics.load,Atomics.store,Atomics.compareExchange;var A;s.wasmBinary&&(A=s.wasmBinary);var j=s.noExitRuntime||!0;typeof WebAssembly!=`object`&&Te(`no native wasm support detected`);var M,ee,N=!1,P;function F(e,t){e||Te(t)}var te=typeof TextDecoder<`u`?new TextDecoder(`utf8`):void 0;function ne(e,t,n){t>>>=0;for(var r=t+n,i=t;e[i]&&!(i>=r);)++i;if(i-t>16&&e.buffer&&te)return te.decode(e.buffer instanceof SharedArrayBuffer?e.slice(t,i):e.subarray(t,i));for(var a=``;t<i;){var o=e[t++];if(!(o&128)){a+=String.fromCharCode(o);continue}var s=e[t++]&63;if((o&224)==192){a+=String.fromCharCode((o&31)<<6|s);continue}var c=e[t++]&63;if(o=(o&240)==224?(o&15)<<12|s<<6|c:(o&7)<<18|s<<12|c<<6|e[t++]&63,o<65536)a+=String.fromCharCode(o);else{var l=o-65536;a+=String.fromCharCode(55296|l>>10,56320|l&1023)}}return a}function ie(e,t){return e>>>=0,e?ne(r(),e,t):``}function ae(e,t,n,r){if(n>>>=0,!(r>0))return 0;for(var i=n,a=n+r-1,o=0;o<e.length;++o){var s=e.charCodeAt(o);if(s>=55296&&s<=57343){var c=e.charCodeAt(++o);s=65536+((s&1023)<<10)|c&1023}if(s<=127){if(n>=a)break;t[n++>>>0]=s}else if(s<=2047){if(n+1>=a)break;t[n++>>>0]=192|s>>6,t[n++>>>0]=128|s&63}else if(s<=65535){if(n+2>=a)break;t[n++>>>0]=224|s>>12,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}else{if(n+3>=a)break;t[n++>>>0]=240|s>>18,t[n++>>>0]=128|s>>12&63,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}}return t[n>>>0]=0,n-i}function oe(e,t,n){return ae(e,r(),t,n)}var I,L,se,R,ce,le;_&&(I=s.buffer);function z(e){I=e,s.HEAP8=L=new Int8Array(e),s.HEAP16=new Int16Array(e),s.HEAP32=R=new Int32Array(e),s.HEAPU8=se=new Uint8Array(e),s.HEAPU16=new Uint16Array(e),s.HEAPU32=ce=new Uint32Array(e),s.HEAPF32=new Float32Array(e),s.HEAPF64=le=new Float64Array(e)}var ue=s.INITIAL_MEMORY||16777216;if(_)M=s.wasmMemory,I=s.buffer;else if(s.wasmMemory)M=s.wasmMemory;else if(M=new WebAssembly.Memory({initial:ue/65536,maximum:65536,shared:!0}),!(M.buffer instanceof SharedArrayBuffer))throw k(`requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag`),g&&k(`(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)`),Error(`bad memory`);M&&(I=M.buffer),ue=I.byteLength,z(I);var de,fe=[],B=[],pe=[];function me(){return j}function he(){if(s.preRun)for(typeof s.preRun==`function`&&(s.preRun=[s.preRun]);s.preRun.length;)ve(s.preRun.shift());Ue(fe)}function ge(){_||Ue(B)}function _e(){if(!_){if(s.postRun)for(typeof s.postRun==`function`&&(s.postRun=[s.postRun]);s.postRun.length;)be(s.postRun.shift());Ue(pe)}}function ve(e){fe.unshift(e)}function ye(e){B.unshift(e)}function be(e){pe.unshift(e)}var V=0,xe=null,Se=null;function Ce(e){V++,s.monitorRunDependencies&&s.monitorRunDependencies(V)}function we(e){if(V--,s.monitorRunDependencies&&s.monitorRunDependencies(V),V==0&&(xe!==null&&(clearInterval(xe),xe=null),Se)){var t=Se;Se=null,t()}}function Te(e){s.onAbort&&s.onAbort(e),e=`Aborted(`+e+`)`,k(e),N=!0,P=1,e+=`. Build with -sASSERTIONS for more info.`;var t=new WebAssembly.RuntimeError(e);throw l(t),t}var Ee=`data:application/octet-stream;base64,`;function De(e){return e.startsWith(Ee)}function Oe(e){return e.startsWith(`file://`)}var H=`tfjs-backend-wasm-threaded-simd.wasm`;De(H)||(H=y(H));function ke(e){try{if(e==H&&A)return new Uint8Array(A);if(S)return S(e);throw`both async and sync fetching of the wasm failed`}catch(e){Te(e)}}function Ae(){if(!A&&(m||h)){if(typeof fetch==`function`&&!Oe(H))return fetch(H,{credentials:`same-origin`}).then(function(e){if(!e.ok)throw`failed to load wasm binary file at '`+H+`'`;return e.arrayBuffer()}).catch(function(){return ke(H)});if(x)return new Promise(function(e,t){x(H,function(t){e(new Uint8Array(t))},t)})}return Promise.resolve().then(function(){return ke(H)})}function je(){var e={env:At,wasi_snapshot_preview1:At};function t(e,t){if(s.asm=e.exports,W(s.asm._emscripten_tls_init),de=s.asm.__indirect_function_table,ye(s.asm.__wasm_call_ctors),ee=t,!_){var n=U.unusedWorkers.length;U.unusedWorkers.forEach(function(e){U.loadWasmModuleToWorker(e,function(){--n||we(`wasm-instantiate`)})})}}_||Ce(`wasm-instantiate`);function n(e){t(e.instance,e.module)}function r(t){return Ae().then(function(t){return WebAssembly.instantiate(t,e)}).then(function(e){return e}).then(t,function(e){k(`failed to asynchronously prepare wasm: `+e),Te(e)})}function i(){return!A&&typeof WebAssembly.instantiateStreaming==`function`&&!De(H)&&!Oe(H)&&!g&&typeof fetch==`function`?fetch(H,{credentials:`same-origin`}).then(function(t){return WebAssembly.instantiateStreaming(t,e).then(n,function(e){return k(`wasm streaming compile failed: `+e),k(`falling back to ArrayBuffer instantiation`),r(n)})}):r(n)}if(s.instantiateWasm)try{return s.instantiateWasm(e,t)}catch(e){k(`Module.instantiateWasm callback failed with error: `+e),l(e)}return i().catch(l),{}}var Me={};function Ne(e){this.name=`ExitStatus`,this.message=`Program terminated with exit(`+e+`)`,this.status=e}function Pe(e){var t=U.pthreads[e];delete U.pthreads[e],t.terminate(),vi(e),U.runningWorkers.splice(U.runningWorkers.indexOf(t),1),t.pthread_ptr=0}function Fe(e){U.pthreads[e].postMessage({cmd:`cancel`})}function Ie(e){var t=U.pthreads[e];F(t),U.returnWorkerToPool(t)}function Le(e){var t=U.getNewWorker();if(!t)return 6;U.runningWorkers.push(t),U.pthreads[e.pthread_ptr]=t,t.pthread_ptr=e.pthread_ptr;var n={cmd:`run`,start_routine:e.startRoutine,arg:e.arg,pthread_ptr:e.pthread_ptr};return t.runPthread=()=>{g&&t.ref(),t.postMessage(n,e.transferList),delete t.runPthread},t.loaded&&t.runPthread(),0}var Re={varargs:void 0,get:function(){return Re.varargs+=4,i()[Re.varargs-4>>>2]},getStr:function(e){return ie(e)}};function ze(e){if(_)return mt(1,1,e);P=e,me()||(U.terminateAllThreads(),s.onExit&&s.onExit(e),N=!0),p(e,new Ne(e))}function Be(e,t){if(P=e,!t&&_)throw Ge(e),`unwind`;ze(e)}var Ve=Be;function He(e){if(e instanceof Ne||e==`unwind`)return P;p(1,e)}var U={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init:function(){_?U.initWorker():U.initMainThread()},initMainThread:function(){for(var e=8;e--;)U.allocateUnusedWorker()},initWorker:function(){j=!1},setExitStatus:function(e){P=e},terminateAllThreads:function(){for(var e of Object.values(U.pthreads))U.returnWorkerToPool(e);for(var e of U.unusedWorkers)e.terminate();U.unusedWorkers=[]},returnWorkerToPool:function(e){var t=e.pthread_ptr;delete U.pthreads[t],U.unusedWorkers.push(e),U.runningWorkers.splice(U.runningWorkers.indexOf(e),1),e.pthread_ptr=0,g&&e.unref(),vi(t)},receiveObjectTransfer:function(e){},threadInitTLS:function(){U.tlsInitFunctions.forEach(e=>e())},loadWasmModuleToWorker:function(t,n){t.onmessage=e=>{var r=e.data,i=r.cmd;if(t.pthread_ptr&&(U.currentProxiedOperationCallerThread=t.pthread_ptr),r.targetThread&&r.targetThread!=li()){var a=U.pthreads[r.targetThread];a?a.postMessage(r,r.transferList):k(`Internal error! Worker sent a message "`+i+`" to target pthread `+r.targetThread+`, but that thread no longer exists!`),U.currentProxiedOperationCallerThread=void 0;return}i===`processProxyingQueue`?G(r.queue):i===`spawnThread`?Le(r):i===`cleanupThread`?Ie(r.thread):i===`killThread`?Pe(r.thread):i===`cancelThread`?Fe(r.thread):i===`loaded`?(t.loaded=!0,g&&t.unref(),n&&n(t),t.runPthread&&t.runPthread()):i===`print`?O(`Thread `+r.threadId+`: `+r.text):i===`printErr`?k(`Thread `+r.threadId+`: `+r.text):i===`alert`?alert(`Thread `+r.threadId+`: `+r.text):r.target===`setimmediate`?t.postMessage(r):i===`callHandler`?s[r.handler](...r.args):i&&k(`worker sent an unknown command `+i),U.currentProxiedOperationCallerThread=void 0},t.onerror=e=>{throw k(`worker sent an error! `+e.filename+`:`+e.lineno+`: `+e.message),e},g&&(t.on(`message`,function(e){t.onmessage({data:e})}),t.on(`error`,function(e){t.onerror(e)}),t.on(`detachedExit`,function(){}));var r=[];for(var i of[`onExit`,`onAbort`,`print`,`printErr`])s.hasOwnProperty(i)&&r.push(i);t.postMessage({cmd:`load`,handlers:r,urlOrBlob:s.mainScriptUrlOrBlob||e,wasmMemory:M,wasmModule:ee})},allocateUnusedWorker:function(){var e,t=y(`tfjs-backend-wasm-threaded-simd.worker.js`);e=new Worker(t),U.unusedWorkers.push(e)},getNewWorker:function(){return U.unusedWorkers.length==0&&(U.allocateUnusedWorker(),U.loadWasmModuleToWorker(U.unusedWorkers[0])),U.unusedWorkers.pop()}};s.PThread=U;function Ue(e){for(;e.length>0;)e.shift()(s)}function We(){var e=li(),t=i()[e+52>>>2],n=t-i()[e+56>>>2];bi(t,n),Si(t)}s.establishStackSpace=We;function Ge(e){if(_)return mt(2,0,e);try{Ve(e)}catch(e){He(e)}}var Ke=[];function qe(e){var t=Ke[e];return t||(e>=Ke.length&&(Ke.length=e+1),Ke[e]=t=de.get(e)),t}function Je(e,t){var n=qe(e)(t);me()?U.setExitStatus(n):yi(n)}s.invokeEntryPoint=Je;function W(e){U.tlsInitFunctions.push(e)}function Ye(e){di(e,!h,1,!m),U.threadInitTLS()}function Xe(e){_?postMessage({cmd:`cleanupThread`,thread:e}):Ie(e)}function Ze(e,t,n,r){return _?mt(3,1,e,t,n,r):Qe(e,t,n,r)}function Qe(e,t,n,r){if(typeof SharedArrayBuffer>`u`)return k(`Current environment does not support SharedArrayBuffer, pthreads are not available!`),6;var i=[];if(_&&i.length===0)return Ze(e,t,n,r);var a={startRoutine:n,pthread_ptr:e,arg:r,transferList:i};return _?(a.cmd=`spawnThread`,postMessage(a,i),0):Le(a)}function $e(){return 65536}var et=!0;function tt(){return et}function G(e){Atomics.store(i(),e>>2,1),li()&&_i(e),Atomics.compareExchange(i(),e>>2,1,0)}s.executeNotifiedProxyingQueue=G;function nt(e,t,n,r){if(e==t)setTimeout(()=>G(r));else if(_)postMessage({targetThread:e,cmd:`processProxyingQueue`,queue:r});else{var i=U.pthreads[e];if(!i)return;i.postMessage({cmd:`processProxyingQueue`,queue:r})}return 1}function rt(e,t,n){return-1}function it(){Te(``)}function at(e){at.shown||={},at.shown[e]||(at.shown[e]=1,g&&(e=`warning: `+e),k(e))}function ot(){g||h||at(`Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread`)}function st(){return Date.now()}function ct(){return 4294901760}function lt(){return ct()}var ut=g?()=>{var e=process.hrtime();return e[0]*1e3+e[1]/1e6}:()=>performance.timeOrigin+performance.now();function dt(e,t,n){r().copyWithin(e>>>0,t>>>0,t+n>>>0)}function ft(){return g?re().cpus().length:navigator.hardwareConcurrency}function pt(e){var t=xi(),n=e();return Si(t),n}function mt(e,t){var n=arguments.length-2,r=arguments;return pt(()=>{for(var i=n,a=Ci(i*8),s=a>>3,c=0;c<n;c++){var l=r[2+c];o()[s+c>>>0]=l}return hi(e,i,a,t)})}var ht=[];function gt(e,t,n){ht.length=t;for(var r=n>>3,i=0;i<t;i++)ht[i]=o()[r+i>>>0];return(e<0?Me[-e-1]:kt[e]).apply(null,ht)}function _t(e){try{return M.grow(e-I.byteLength+65535>>>16),z(M.buffer),1}catch{}}function vt(e){var t=r().length;if(e>>>=0,e<=t)return!1;var n=ct();if(e>n)return!1;let i=(e,t)=>e+(t-e%t)%t;for(var a=1;a<=4;a*=2){var o=t*(1+.2/a);if(o=Math.min(o,e+100663296),_t(Math.min(n,i(Math.max(e,o),65536))))return!0}return!1}function yt(){throw`unwind`}function bt(e){return _?mt(4,1,e):52}function xt(e,t,n,r,i){return _?mt(5,1,e,t,n,r,i):70}var St=[null,[],[]];function Ct(e,t){var n=St[e];t===0||t===10?((e===1?O:k)(ne(n,0)),n.length=0):n.push(t)}function wt(e,t,n,i){if(_)return mt(6,1,e,t,n,i);for(var o=0,s=0;s<n;s++){var c=a()[t>>>2],l=a()[t+4>>>2];t+=8;for(var u=0;u<l;u++)Ct(e,r()[c+u>>>0]);o+=l}return a()[i>>>2]=o,0}function Tt(e){return s[`_`+e]}function Et(e,t){n().set(e,t>>>0)}function Dt(e,t,n,r,i){var a={string:e=>{var t=0;if(e!=null&&e!==0){var n=(e.length<<2)+1;t=Ci(n),oe(e,t,n)}return t},array:e=>{var t=Ci(e.length);return Et(e,t),t}};function o(e){return t===`string`?ie(e):t===`boolean`?!!e:e}var s=Tt(e),c=[],l=0;if(r)for(var u=0;u<r.length;u++){var d=a[n[u]];d?(l===0&&(l=xi()),c[u]=d(r[u])):c[u]=r[u]}var f=s.apply(null,c);function p(e){return l!==0&&Si(l),o(e)}return f=p(f),f}function Ot(e,t,n,r){n||=[];var i=n.every(e=>e===`number`||e===`boolean`);return t!==`string`&&i&&!r?Tt(e):function(){return Dt(e,t,n,arguments,r)}}U.init();var kt=[null,ze,Ge,Ze,bt,xt,wt],At={__emscripten_init_main_thread_js:Ye,__emscripten_thread_cleanup:Xe,__pthread_create_js:Qe,_emscripten_default_pthread_stack_size:$e,_emscripten_get_now_is_monotonic:tt,_emscripten_notify_task_queue:nt,_emscripten_set_offscreencanvas_size:rt,abort:it,emscripten_check_blocking_allowed:ot,emscripten_date_now:st,emscripten_get_heap_max:lt,emscripten_get_now:ut,emscripten_memcpy_big:dt,emscripten_num_logical_cores:ft,emscripten_receive_on_main_thread_js:gt,emscripten_resize_heap:vt,emscripten_unwind_to_js_event_loop:yt,exit:Ve,fd_close:bt,fd_seek:xt,fd_write:wt,memory:M||s.wasmMemory};je();var jt=s.___wasm_call_ctors=function(){return(jt=s.___wasm_call_ctors=s.asm.__wasm_call_ctors).apply(null,arguments)},Mt=s._init=function(){return(Mt=s._init=s.asm.init).apply(null,arguments)},Nt=s._init_with_threads_count=function(){return(Nt=s._init_with_threads_count=s.asm.init_with_threads_count).apply(null,arguments)},Pt=s._get_threads_count=function(){return(Pt=s._get_threads_count=s.asm.get_threads_count).apply(null,arguments)},Ft=s._register_tensor=function(){return(Ft=s._register_tensor=s.asm.register_tensor).apply(null,arguments)},It=s._dispose_data=function(){return(It=s._dispose_data=s.asm.dispose_data).apply(null,arguments)},Lt=s._dispose=function(){return(Lt=s._dispose=s.asm.dispose).apply(null,arguments)},Rt=s._Abs=function(){return(Rt=s._Abs=s.asm.Abs).apply(null,arguments)},zt=s._Acos=function(){return(zt=s._Acos=s.asm.Acos).apply(null,arguments)},Bt=s._Acosh=function(){return(Bt=s._Acosh=s.asm.Acosh).apply(null,arguments)},Vt=s._Add=function(){return(Vt=s._Add=s.asm.Add).apply(null,arguments)},Ht=s._AddN=function(){return(Ht=s._AddN=s.asm.AddN).apply(null,arguments)},Ut=s._All=function(){return(Ut=s._All=s.asm.All).apply(null,arguments)},Wt=s._Any=function(){return(Wt=s._Any=s.asm.Any).apply(null,arguments)},Gt=s._ArgMax=function(){return(Gt=s._ArgMax=s.asm.ArgMax).apply(null,arguments)},Kt=s._ArgMin=function(){return(Kt=s._ArgMin=s.asm.ArgMin).apply(null,arguments)},qt=s._Asin=function(){return(qt=s._Asin=s.asm.Asin).apply(null,arguments)},Jt=s._Asinh=function(){return(Jt=s._Asinh=s.asm.Asinh).apply(null,arguments)},Yt=s._Atan=function(){return(Yt=s._Atan=s.asm.Atan).apply(null,arguments)},Xt=s._Atan2=function(){return(Xt=s._Atan2=s.asm.Atan2).apply(null,arguments)},Zt=s._Atanh=function(){return(Zt=s._Atanh=s.asm.Atanh).apply(null,arguments)},Qt=s._AvgPool=function(){return(Qt=s._AvgPool=s.asm.AvgPool).apply(null,arguments)},$t=s._AvgPool3D=function(){return($t=s._AvgPool3D=s.asm.AvgPool3D).apply(null,arguments)},en=s._AvgPool3DGrad=function(){return(en=s._AvgPool3DGrad=s.asm.AvgPool3DGrad).apply(null,arguments)},tn=s._AvgPoolGrad=function(){return(tn=s._AvgPoolGrad=s.asm.AvgPoolGrad).apply(null,arguments)},nn=s._BatchMatMul=function(){return(nn=s._BatchMatMul=s.asm.BatchMatMul).apply(null,arguments)},rn=s._Bincount=function(){return(rn=s._Bincount=s.asm.Bincount).apply(null,arguments)},an=s._BitwiseAnd=function(){return(an=s._BitwiseAnd=s.asm.BitwiseAnd).apply(null,arguments)},on=s._Ceil=function(){return(on=s._Ceil=s.asm.Ceil).apply(null,arguments)},sn=s._ClipByValue=function(){return(sn=s._ClipByValue=s.asm.ClipByValue).apply(null,arguments)},K=s._Conv2D=function(){return(K=s._Conv2D=s.asm.Conv2D).apply(null,arguments)},cn=s._Conv2DBackpropInput=function(){return(cn=s._Conv2DBackpropInput=s.asm.Conv2DBackpropInput).apply(null,arguments)},ln=s._Conv3D=function(){return(ln=s._Conv3D=s.asm.Conv3D).apply(null,arguments)},un=s._Conv3DBackpropFilterV2=function(){return(un=s._Conv3DBackpropFilterV2=s.asm.Conv3DBackpropFilterV2).apply(null,arguments)},dn=s._Conv3DBackpropInputV2=function(){return(dn=s._Conv3DBackpropInputV2=s.asm.Conv3DBackpropInputV2).apply(null,arguments)},fn=s._Cos=function(){return(fn=s._Cos=s.asm.Cos).apply(null,arguments)},pn=s._Cosh=function(){return(pn=s._Cosh=s.asm.Cosh).apply(null,arguments)},mn=s._CropAndResize=function(){return(mn=s._CropAndResize=s.asm.CropAndResize).apply(null,arguments)},hn=s._Cumprod=function(){return(hn=s._Cumprod=s.asm.Cumprod).apply(null,arguments)},gn=s._Cumsum=function(){return(gn=s._Cumsum=s.asm.Cumsum).apply(null,arguments)},_n=s._DenseBincount=function(){return(_n=s._DenseBincount=s.asm.DenseBincount).apply(null,arguments)},vn=s._DepthToSpace=function(){return(vn=s._DepthToSpace=s.asm.DepthToSpace).apply(null,arguments)},yn=s._DepthwiseConv2dNative=function(){return(yn=s._DepthwiseConv2dNative=s.asm.DepthwiseConv2dNative).apply(null,arguments)},bn=s._Diag=function(){return(bn=s._Diag=s.asm.Diag).apply(null,arguments)},xn=s._Dilation2D=function(){return(xn=s._Dilation2D=s.asm.Dilation2D).apply(null,arguments)},Sn=s._Dilation2DBackpropFilter=function(){return(Sn=s._Dilation2DBackpropFilter=s.asm.Dilation2DBackpropFilter).apply(null,arguments)},Cn=s._Dilation2DBackpropInput=function(){return(Cn=s._Dilation2DBackpropInput=s.asm.Dilation2DBackpropInput).apply(null,arguments)},wn=s._Elu=function(){return(wn=s._Elu=s.asm.Elu).apply(null,arguments)},Tn=s._EluGrad=function(){return(Tn=s._EluGrad=s.asm.EluGrad).apply(null,arguments)},q=s._Equal=function(){return(q=s._Equal=s.asm.Equal).apply(null,arguments)},En=s._Erf=function(){return(En=s._Erf=s.asm.Erf).apply(null,arguments)},Dn=s._Exp=function(){return(Dn=s._Exp=s.asm.Exp).apply(null,arguments)},On=s._Expm1=function(){return(On=s._Expm1=s.asm.Expm1).apply(null,arguments)},kn=s._FlipLeftRight=function(){return(kn=s._FlipLeftRight=s.asm.FlipLeftRight).apply(null,arguments)},An=s._Floor=function(){return(An=s._Floor=s.asm.Floor).apply(null,arguments)},jn=s._FloorDiv=function(){return(jn=s._FloorDiv=s.asm.FloorDiv).apply(null,arguments)},Mn=s._FusedBatchNorm=function(){return(Mn=s._FusedBatchNorm=s.asm.FusedBatchNorm).apply(null,arguments)},Nn=s._FusedConv2D=function(){return(Nn=s._FusedConv2D=s.asm.FusedConv2D).apply(null,arguments)},Pn=s._FusedDepthwiseConv2D=function(){return(Pn=s._FusedDepthwiseConv2D=s.asm.FusedDepthwiseConv2D).apply(null,arguments)},Fn=s._Gather=function(){return(Fn=s._Gather=s.asm.Gather).apply(null,arguments)},In=s._GatherNd=function(){return(In=s._GatherNd=s.asm.GatherNd).apply(null,arguments)},Ln=s._Greater=function(){return(Ln=s._Greater=s.asm.Greater).apply(null,arguments)},Rn=s._GreaterEqual=function(){return(Rn=s._GreaterEqual=s.asm.GreaterEqual).apply(null,arguments)},zn=s._IsFinite=function(){return(zn=s._IsFinite=s.asm.IsFinite).apply(null,arguments)},Bn=s._IsInf=function(){return(Bn=s._IsInf=s.asm.IsInf).apply(null,arguments)},Vn=s._IsNan=function(){return(Vn=s._IsNan=s.asm.IsNan).apply(null,arguments)},Hn=s._LRN=function(){return(Hn=s._LRN=s.asm.LRN).apply(null,arguments)},Un=s._LRNGrad=function(){return(Un=s._LRNGrad=s.asm.LRNGrad).apply(null,arguments)},Wn=s._LeakyRelu=function(){return(Wn=s._LeakyRelu=s.asm.LeakyRelu).apply(null,arguments)},Gn=s._Less=function(){return(Gn=s._Less=s.asm.Less).apply(null,arguments)},Kn=s._LessEqual=function(){return(Kn=s._LessEqual=s.asm.LessEqual).apply(null,arguments)},qn=s._LinSpace=function(){return(qn=s._LinSpace=s.asm.LinSpace).apply(null,arguments)},Jn=s._Log=function(){return(Jn=s._Log=s.asm.Log).apply(null,arguments)},Yn=s._Log1p=function(){return(Yn=s._Log1p=s.asm.Log1p).apply(null,arguments)},Xn=s._LogicalAnd=function(){return(Xn=s._LogicalAnd=s.asm.LogicalAnd).apply(null,arguments)},Zn=s._LogicalNot=function(){return(Zn=s._LogicalNot=s.asm.LogicalNot).apply(null,arguments)},Qn=s._LogicalOr=function(){return(Qn=s._LogicalOr=s.asm.LogicalOr).apply(null,arguments)},$n=s._LogicalXor=function(){return($n=s._LogicalXor=s.asm.LogicalXor).apply(null,arguments)},er=s._Max=function(){return(er=s._Max=s.asm.Max).apply(null,arguments)},J=s._MaxPool=function(){return(J=s._MaxPool=s.asm.MaxPool).apply(null,arguments)},tr=s._MaxPool3D=function(){return(tr=s._MaxPool3D=s.asm.MaxPool3D).apply(null,arguments)},nr=s._MaxPool3DGrad=function(){return(nr=s._MaxPool3DGrad=s.asm.MaxPool3DGrad).apply(null,arguments)},rr=s._MaxPoolGrad=function(){return(rr=s._MaxPoolGrad=s.asm.MaxPoolGrad).apply(null,arguments)},ir=s._MaxPoolWithArgmax=function(){return(ir=s._MaxPoolWithArgmax=s.asm.MaxPoolWithArgmax).apply(null,arguments)},ar=s._Maximum=function(){return(ar=s._Maximum=s.asm.Maximum).apply(null,arguments)},Y=s._Mean=function(){return(Y=s._Mean=s.asm.Mean).apply(null,arguments)},or=s._Min=function(){return(or=s._Min=s.asm.Min).apply(null,arguments)},sr=s._Minimum=function(){return(sr=s._Minimum=s.asm.Minimum).apply(null,arguments)},cr=s._MirrorPad=function(){return(cr=s._MirrorPad=s.asm.MirrorPad).apply(null,arguments)},X=s._Mod=function(){return(X=s._Mod=s.asm.Mod).apply(null,arguments)},lr=s._Multinomial=function(){return(lr=s._Multinomial=s.asm.Multinomial).apply(null,arguments)},ur=s._Multiply=function(){return(ur=s._Multiply=s.asm.Multiply).apply(null,arguments)},dr=s._Neg=function(){return(dr=s._Neg=s.asm.Neg).apply(null,arguments)},fr=s._NonMaxSuppressionV3=function(){return(fr=s._NonMaxSuppressionV3=s.asm.NonMaxSuppressionV3).apply(null,arguments)},pr=s._NonMaxSuppressionV4=function(){return(pr=s._NonMaxSuppressionV4=s.asm.NonMaxSuppressionV4).apply(null,arguments)},mr=s._NonMaxSuppressionV5=function(){return(mr=s._NonMaxSuppressionV5=s.asm.NonMaxSuppressionV5).apply(null,arguments)},hr=s._NotEqual=function(){return(hr=s._NotEqual=s.asm.NotEqual).apply(null,arguments)},gr=s._OneHot=function(){return(gr=s._OneHot=s.asm.OneHot).apply(null,arguments)},_r=s._PadV2=function(){return(_r=s._PadV2=s.asm.PadV2).apply(null,arguments)},Z=s._Pow=function(){return(Z=s._Pow=s.asm.Pow).apply(null,arguments)},vr=s._Prelu=function(){return(vr=s._Prelu=s.asm.Prelu).apply(null,arguments)},yr=s._Prod=function(){return(yr=s._Prod=s.asm.Prod).apply(null,arguments)},br=s._RealDiv=function(){return(br=s._RealDiv=s.asm.RealDiv).apply(null,arguments)},xr=s._Reciprocal=function(){return(xr=s._Reciprocal=s.asm.Reciprocal).apply(null,arguments)},Sr=s._Relu=function(){return(Sr=s._Relu=s.asm.Relu).apply(null,arguments)},Cr=s._Relu6=function(){return(Cr=s._Relu6=s.asm.Relu6).apply(null,arguments)},wr=s._ResizeBilinear=function(){return(wr=s._ResizeBilinear=s.asm.ResizeBilinear).apply(null,arguments)},Tr=s._ResizeBilinearGrad=function(){return(Tr=s._ResizeBilinearGrad=s.asm.ResizeBilinearGrad).apply(null,arguments)},Er=s._ResizeNearestNeighbor=function(){return(Er=s._ResizeNearestNeighbor=s.asm.ResizeNearestNeighbor).apply(null,arguments)},Dr=s._ResizeNearestNeighborGrad=function(){return(Dr=s._ResizeNearestNeighborGrad=s.asm.ResizeNearestNeighborGrad).apply(null,arguments)},Or=s._Reverse=function(){return(Or=s._Reverse=s.asm.Reverse).apply(null,arguments)},kr=s._RotateWithOffset=function(){return(kr=s._RotateWithOffset=s.asm.RotateWithOffset).apply(null,arguments)},Ar=s._Round=function(){return(Ar=s._Round=s.asm.Round).apply(null,arguments)},jr=s._Rsqrt=function(){return(jr=s._Rsqrt=s.asm.Rsqrt).apply(null,arguments)},Mr=s._ScatterNd=function(){return(Mr=s._ScatterNd=s.asm.ScatterNd).apply(null,arguments)},Nr=s._SearchSorted=function(){return(Nr=s._SearchSorted=s.asm.SearchSorted).apply(null,arguments)},Pr=s._SelectV2=function(){return(Pr=s._SelectV2=s.asm.SelectV2).apply(null,arguments)},Fr=s._Selu=function(){return(Fr=s._Selu=s.asm.Selu).apply(null,arguments)},Ir=s._Sigmoid=function(){return(Ir=s._Sigmoid=s.asm.Sigmoid).apply(null,arguments)},Lr=s._Sign=function(){return(Lr=s._Sign=s.asm.Sign).apply(null,arguments)},Rr=s._Sin=function(){return(Rr=s._Sin=s.asm.Sin).apply(null,arguments)},zr=s._Sinh=function(){return(zr=s._Sinh=s.asm.Sinh).apply(null,arguments)},Br=s._Softmax=function(){return(Br=s._Softmax=s.asm.Softmax).apply(null,arguments)},Vr=s._Softplus=function(){return(Vr=s._Softplus=s.asm.Softplus).apply(null,arguments)},Hr=s._SparseFillEmptyRows=function(){return(Hr=s._SparseFillEmptyRows=s.asm.SparseFillEmptyRows).apply(null,arguments)},Ur=s._SparseReshape=function(){return(Ur=s._SparseReshape=s.asm.SparseReshape).apply(null,arguments)},Wr=s._SparseSegmentReduction=function(){return(Wr=s._SparseSegmentReduction=s.asm.SparseSegmentReduction).apply(null,arguments)},Gr=s._SparseToDense=function(){return(Gr=s._SparseToDense=s.asm.SparseToDense).apply(null,arguments)},Kr=s._Sqrt=function(){return(Kr=s._Sqrt=s.asm.Sqrt).apply(null,arguments)},qr=s._Square=function(){return(qr=s._Square=s.asm.Square).apply(null,arguments)},Jr=s._SquaredDifference=function(){return(Jr=s._SquaredDifference=s.asm.SquaredDifference).apply(null,arguments)},Yr=s._Step=function(){return(Yr=s._Step=s.asm.Step).apply(null,arguments)},Xr=s._StridedSlice=function(){return(Xr=s._StridedSlice=s.asm.StridedSlice).apply(null,arguments)},Zr=s._Sub=function(){return(Zr=s._Sub=s.asm.Sub).apply(null,arguments)},Qr=s._Sum=function(){return(Qr=s._Sum=s.asm.Sum).apply(null,arguments)},$r=s._Tan=function(){return($r=s._Tan=s.asm.Tan).apply(null,arguments)},Q=s._Tanh=function(){return(Q=s._Tanh=s.asm.Tanh).apply(null,arguments)},ei=s._TensorScatterUpdate=function(){return(ei=s._TensorScatterUpdate=s.asm.TensorScatterUpdate).apply(null,arguments)},ti=s._Tile=function(){return(ti=s._Tile=s.asm.Tile).apply(null,arguments)},ni=s._TopK=function(){return(ni=s._TopK=s.asm.TopK).apply(null,arguments)},ri=s._Transform=function(){return(ri=s._Transform=s.asm.Transform).apply(null,arguments)},ii=s._Transpose=function(){return(ii=s._Transpose=s.asm.Transpose).apply(null,arguments)},ai=s.__FusedMatMul=function(){return(ai=s.__FusedMatMul=s.asm._FusedMatMul).apply(null,arguments)},oi=s._malloc=function(){return(oi=s._malloc=s.asm.malloc).apply(null,arguments)},si=s._free=function(){return(si=s._free=s.asm.free).apply(null,arguments)},ci=s.__emscripten_tls_init=function(){return(ci=s.__emscripten_tls_init=s.asm._emscripten_tls_init).apply(null,arguments)},li=s._pthread_self=function(){return(li=s._pthread_self=s.asm.pthread_self).apply(null,arguments)},ui=s.___errno_location=function(){return(ui=s.___errno_location=s.asm.__errno_location).apply(null,arguments)},di=s.__emscripten_thread_init=function(){return(di=s.__emscripten_thread_init=s.asm._emscripten_thread_init).apply(null,arguments)},fi=s.__emscripten_thread_crashed=function(){return(fi=s.__emscripten_thread_crashed=s.asm._emscripten_thread_crashed).apply(null,arguments)},pi=s._emscripten_main_thread_process_queued_calls=function(){return(pi=s._emscripten_main_thread_process_queued_calls=s.asm.emscripten_main_thread_process_queued_calls).apply(null,arguments)},mi=s._emscripten_main_browser_thread_id=function(){return(mi=s._emscripten_main_browser_thread_id=s.asm.emscripten_main_browser_thread_id).apply(null,arguments)},hi=s._emscripten_run_in_main_runtime_thread_js=function(){return(hi=s._emscripten_run_in_main_runtime_thread_js=s.asm.emscripten_run_in_main_runtime_thread_js).apply(null,arguments)},gi=s._emscripten_dispatch_to_thread_=function(){return(gi=s._emscripten_dispatch_to_thread_=s.asm.emscripten_dispatch_to_thread_).apply(null,arguments)},_i=s.__emscripten_proxy_execute_task_queue=function(){return(_i=s.__emscripten_proxy_execute_task_queue=s.asm._emscripten_proxy_execute_task_queue).apply(null,arguments)},vi=s.__emscripten_thread_free_data=function(){return(vi=s.__emscripten_thread_free_data=s.asm._emscripten_thread_free_data).apply(null,arguments)},yi=s.__emscripten_thread_exit=function(){return(yi=s.__emscripten_thread_exit=s.asm._emscripten_thread_exit).apply(null,arguments)},bi=s._emscripten_stack_set_limits=function(){return(bi=s._emscripten_stack_set_limits=s.asm.emscripten_stack_set_limits).apply(null,arguments)},xi=s.stackSave=function(){return(xi=s.stackSave=s.asm.stackSave).apply(null,arguments)},Si=s.stackRestore=function(){return(Si=s.stackRestore=s.asm.stackRestore).apply(null,arguments)},Ci=s.stackAlloc=function(){return(Ci=s.stackAlloc=s.asm.stackAlloc).apply(null,arguments)},wi=s.dynCall_iijjiiii=function(){return(wi=s.dynCall_iijjiiii=s.asm.dynCall_iijjiiii).apply(null,arguments)},Ti=s.dynCall_jiji=function(){return(Ti=s.dynCall_jiji=s.asm.dynCall_jiji).apply(null,arguments)};s.keepRuntimeAlive=me,s.wasmMemory=M,s.cwrap=Ot,s.ExitStatus=Ne,s.PThread=U;var Ei;Se=function e(){Ei||Di(),Ei||(Se=e)};function Di(e){if(e||=f,V>0)return;if(_){c(s),ge(),startWorker(s);return}if(he(),V>0)return;function t(){Ei||(Ei=!0,s.calledRun=!0,!N&&(ge(),c(s),s.onRuntimeInitialized&&s.onRuntimeInitialized(),_e()))}s.setStatus?(s.setStatus(`Running...`),setTimeout(function(){setTimeout(function(){s.setStatus(``)},1),t()},1)):t()}if(s.preInit)for(typeof s.preInit==`function`&&(s.preInit=[s.preInit]);s.preInit.length>0;)s.preInit.pop()();Di();var $;u&&($={uncaughtException:process.listeners(`uncaughtException`).filter(function(e){return!u.uncaughtException.indexOf(e)>-1}),unhandledRejection:process.listeners(`unhandledRejection`).filter(function(e){return!u.unhandledRejection.indexOf(e)>-1})});var Oi;if(typeof WasmBackendModule<`u`)Oi=WasmBackendModule;else if(t!==void 0)Oi=t;else throw Error(`Could not find wasm module in post.js`);if($){var ki=Oi._dispose;Oi._dispose=function(){ki(),$.uncaughtException.forEach(function(e){process.removeListener(`uncaughtException`,e)}),$.unhandledRejection.forEach(function(e){process.removeListener(`unhandledRejection`,e)})}}return t.ready})})();typeof e==`object`&&typeof t==`object`?t.exports=n:typeof define==`function`&&define.amd?define([],function(){return n}):typeof e==`object`&&(e.WasmBackendModuleThreadedSimd=n)})),Zu=/* @__PURE__ */ In(((e,t)=>{t.exports.wasmWorkerContents=`"use strict";var Module={};var ENVIRONMENT_IS_NODE=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string";if(ENVIRONMENT_IS_NODE){var nodeWorkerThreads=require("worker_threads");var parentPort=nodeWorkerThreads.parentPort;parentPort.on("message",data=>onmessage({data:data}));var fs=require("fs");Object.assign(global,{self:global,require:require,Module:Module,location:{href:__filename},Worker:nodeWorkerThreads.Worker,importScripts:function(f){(0,eval)(fs.readFileSync(f,"utf8")+"//# sourceURL="+f)},postMessage:function(msg){parentPort.postMessage(msg)},performance:global.performance||{now:function(){return Date.now()}}})}var initializedJS=false;var pendingNotifiedProxyingQueues=[];function threadPrintErr(){var text=Array.prototype.slice.call(arguments).join(" ");if(ENVIRONMENT_IS_NODE){fs.writeSync(2,text+"
");return}console.error(text)}function threadAlert(){var text=Array.prototype.slice.call(arguments).join(" ");postMessage({cmd:"alert",text:text,threadId:Module["_pthread_self"]()})}var err=threadPrintErr;self.alert=threadAlert;Module["instantiateWasm"]=(info,receiveInstance)=>{var instance=new WebAssembly.Instance(Module["wasmModule"],info);receiveInstance(instance);Module["wasmModule"]=null;return instance.exports};self.onunhandledrejection=e=>{throw e.reason??e};self.startWorker=instance=>{Module=instance;postMessage({"cmd":"loaded"})};self.onmessage=e=>{try{if(e.data.cmd==="load"){Module["wasmModule"]=e.data.wasmModule;for(const handler of e.data.handlers){Module[handler]=function(){postMessage({cmd:"callHandler",handler:handler,args:[...arguments]})}}Module["wasmMemory"]=e.data.wasmMemory;Module["buffer"]=Module["wasmMemory"].buffer;Module["ENVIRONMENT_IS_PTHREAD"]=true;if(typeof e.data.urlOrBlob=="string"){importScripts(e.data.urlOrBlob)}else{var objectUrl=URL.createObjectURL(e.data.urlOrBlob);importScripts(objectUrl);URL.revokeObjectURL(objectUrl)}WasmBackendModuleThreadedSimd(Module)}else if(e.data.cmd==="run"){Module["__emscripten_thread_init"](e.data.pthread_ptr,0,0,1);Module["establishStackSpace"]();Module["PThread"].receiveObjectTransfer(e.data);Module["PThread"].threadInitTLS();if(!initializedJS){pendingNotifiedProxyingQueues.forEach(queue=>{Module["executeNotifiedProxyingQueue"](queue)});pendingNotifiedProxyingQueues=[];initializedJS=true}try{Module["invokeEntryPoint"](e.data.start_routine,e.data.arg)}catch(ex){if(ex!="unwind"){if(ex instanceof Module["ExitStatus"]){if(Module["keepRuntimeAlive"]()){}else{Module["__emscripten_thread_exit"](ex.status)}}else{throw ex}}}}else if(e.data.cmd==="cancel"){if(Module["_pthread_self"]()){Module["__emscripten_thread_exit"](-1)}}else if(e.data.target==="setimmediate"){}else if(e.data.cmd==="processProxyingQueue"){if(initializedJS){Module["executeNotifiedProxyingQueue"](e.data.queue)}else{pendingNotifiedProxyingQueues.push(e.data.queue)}}else if(e.data.cmd){err("worker.js received unknown command "+e.data.cmd);err(e.data)}}catch(ex){if(Module["__emscripten_thread_crashed"]){Module["__emscripten_thread_crashed"]()}throw ex}};`})),Qu=/* @__PURE__ */ In(((e,t)=>{var n=(()=>{var e=typeof document<`u`&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<`u`&&(e||=__filename),(function(t){t||={};var n=t===void 0?{}:t,r,i;n.ready=new Promise(function(e,t){r=e,i=t});var a;typeof process<`u`&&process.listeners&&(a={uncaughtException:process.listeners(`uncaughtException`),unhandledRejection:process.listeners(`unhandledRejection`)});var o=Object.assign({},n),s=[],c=typeof window==`object`,l=typeof importScripts==`function`,u=typeof process==`object`&&typeof process.versions==`object`&&typeof process.versions.node==`string`,d=``;function f(e){return n.locateFile?n.locateFile(e,d):d+e}var p,m,h;if(u){var g=re(),_=re();d=l?_.dirname(d)+`/`:__dirname+`/`,p=(e,t)=>(e=fe(e)?new URL(e):_.normalize(e),g.readFileSync(e,t?void 0:`utf8`)),h=e=>{var t=p(e,!0);return t.buffer||(t=new Uint8Array(t)),t},m=(e,t,n)=>{e=fe(e)?new URL(e):_.normalize(e),g.readFile(e,function(e,r){e?n(e):t(r.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,`/`),s=process.argv.slice(2),process.on(`uncaughtException`,function(e){if(!(e instanceof ge))throw e}),process.on(`unhandledRejection`,function(e){throw e}),n.inspect=function(){return`[Emscripten Module object]`}}else(c||l)&&(l?d=self.location.href:typeof document<`u`&&document.currentScript&&(d=document.currentScript.src),e&&(d=e),d=d.indexOf(`blob:`)===0?``:d.substr(0,d.replace(/[?#].*/,``).lastIndexOf(`/`)+1),p=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.send(null),t.responseText},l&&(h=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.responseType=`arraybuffer`,t.send(null),new Uint8Array(t.response)}),m=(e,t,n)=>{var r=new XMLHttpRequest;r.open(`GET`,e,!0),r.responseType=`arraybuffer`,r.onload=()=>{if(r.status==200||r.status==0&&r.response){t(r.response);return}n()},r.onerror=n,r.send(null)});var v=n.print||console.log.bind(console),y=n.printErr||console.warn.bind(console);Object.assign(n,o),o=null,n.arguments&&(s=n.arguments),n.thisProgram&&n.thisProgram,n.quit&&n.quit;var b;n.wasmBinary&&(b=n.wasmBinary),n.noExitRuntime,typeof WebAssembly!=`object`&&z(`no native wasm support detected`);var x,S=!1,C=typeof TextDecoder<`u`?new TextDecoder(`utf8`):void 0;function w(e,t,n){t>>>=0;for(var r=t+n,i=t;e[i]&&!(i>=r);)++i;if(i-t>16&&e.buffer&&C)return C.decode(e.subarray(t,i));for(var a=``;t<i;){var o=e[t++];if(!(o&128)){a+=String.fromCharCode(o);continue}var s=e[t++]&63;if((o&224)==192){a+=String.fromCharCode((o&31)<<6|s);continue}var c=e[t++]&63;if(o=(o&240)==224?(o&15)<<12|s<<6|c:(o&7)<<18|s<<12|c<<6|e[t++]&63,o<65536)a+=String.fromCharCode(o);else{var l=o-65536;a+=String.fromCharCode(55296|l>>10,56320|l&1023)}}return a}function T(e,t){return e>>>=0,e?w(A,e,t):``}function E(e,t,n,r){if(n>>>=0,!(r>0))return 0;for(var i=n,a=n+r-1,o=0;o<e.length;++o){var s=e.charCodeAt(o);if(s>=55296&&s<=57343){var c=e.charCodeAt(++o);s=65536+((s&1023)<<10)|c&1023}if(s<=127){if(n>=a)break;t[n++>>>0]=s}else if(s<=2047){if(n+1>=a)break;t[n++>>>0]=192|s>>6,t[n++>>>0]=128|s&63}else if(s<=65535){if(n+2>=a)break;t[n++>>>0]=224|s>>12,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}else{if(n+3>=a)break;t[n++>>>0]=240|s>>18,t[n++>>>0]=128|s>>12&63,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}}return t[n>>>0]=0,n-i}function D(e,t,n){return E(e,A,t,n)}var O,k,A,j,M;function ee(e){O=e,n.HEAP8=k=new Int8Array(e),n.HEAP16=new Int16Array(e),n.HEAP32=j=new Int32Array(e),n.HEAPU8=A=new Uint8Array(e),n.HEAPU16=new Uint16Array(e),n.HEAPU32=M=new Uint32Array(e),n.HEAPF32=new Float32Array(e),n.HEAPF64=new Float64Array(e)}n.INITIAL_MEMORY;var N=[],P=[],F=[];function te(){if(n.preRun)for(typeof n.preRun==`function`&&(n.preRun=[n.preRun]);n.preRun.length;)ae(n.preRun.shift());_e(N)}function ne(){_e(P)}function ie(){if(n.postRun)for(typeof n.postRun==`function`&&(n.postRun=[n.postRun]);n.postRun.length;)I(n.postRun.shift());_e(F)}function ae(e){N.unshift(e)}function oe(e){P.unshift(e)}function I(e){F.unshift(e)}var L=0,se=null,R=null;function ce(e){L++,n.monitorRunDependencies&&n.monitorRunDependencies(L)}function le(e){if(L--,n.monitorRunDependencies&&n.monitorRunDependencies(L),L==0&&(se!==null&&(clearInterval(se),se=null),R)){var t=R;R=null,t()}}function z(e){n.onAbort&&n.onAbort(e),e=`Aborted(`+e+`)`,y(e),S=!0,e+=`. Build with -sASSERTIONS for more info.`;var t=new WebAssembly.RuntimeError(e);throw i(t),t}var ue=`data:application/octet-stream;base64,`;function de(e){return e.startsWith(ue)}function fe(e){return e.startsWith(`file://`)}var B=`tfjs-backend-wasm.wasm`;de(B)||(B=f(B));function pe(e){try{if(e==B&&b)return new Uint8Array(b);if(h)return h(e);throw`both async and sync fetching of the wasm failed`}catch(e){z(e)}}function me(){if(!b&&(c||l)){if(typeof fetch==`function`&&!fe(B))return fetch(B,{credentials:`same-origin`}).then(function(e){if(!e.ok)throw`failed to load wasm binary file at '`+B+`'`;return e.arrayBuffer()}).catch(function(){return pe(B)});if(m)return new Promise(function(e,t){m(B,function(t){e(new Uint8Array(t))},t)})}return Promise.resolve().then(function(){return pe(B)})}function he(){var e={env:Me,wasi_snapshot_preview1:Me};function t(e,t){n.asm=e.exports,x=n.asm.memory,ee(x.buffer),n.asm.__indirect_function_table,oe(n.asm.__wasm_call_ctors),le(`wasm-instantiate`)}ce(`wasm-instantiate`);function r(e){t(e.instance)}function a(t){return me().then(function(t){return WebAssembly.instantiate(t,e)}).then(function(e){return e}).then(t,function(e){y(`failed to asynchronously prepare wasm: `+e),z(e)})}function o(){return!b&&typeof WebAssembly.instantiateStreaming==`function`&&!de(B)&&!fe(B)&&!u&&typeof fetch==`function`?fetch(B,{credentials:`same-origin`}).then(function(t){return WebAssembly.instantiateStreaming(t,e).then(r,function(e){return y(`wasm streaming compile failed: `+e),y(`falling back to ArrayBuffer instantiation`),a(r)})}):a(r)}if(n.instantiateWasm)try{return n.instantiateWasm(e,t)}catch(e){y(`Module.instantiateWasm callback failed with error: `+e),i(e)}return o().catch(i),{}}function ge(e){this.name=`ExitStatus`,this.message=`Program terminated with exit(`+e+`)`,this.status=e}function _e(e){for(;e.length>0;)e.shift()(n)}function ve(){z(``)}function ye(){return 4294901760}function be(){return ye()}function V(e,t,n){A.copyWithin(e>>>0,t>>>0,t+n>>>0)}function xe(e){try{return x.grow(e-O.byteLength+65535>>>16),ee(x.buffer),1}catch{}}function Se(e){var t=A.length;e>>>=0;var n=ye();if(e>n)return!1;let r=(e,t)=>e+(t-e%t)%t;for(var i=1;i<=4;i*=2){var a=t*(1+.2/i);if(a=Math.min(a,e+100663296),xe(Math.min(n,r(Math.max(e,a),65536))))return!0}return!1}var Ce={varargs:void 0,get:function(){return Ce.varargs+=4,j[Ce.varargs-4>>>2]},getStr:function(e){return T(e)}};function we(e){return 52}function Te(e,t,n,r,i){return 70}var Ee=[null,[],[]];function De(e,t){var n=Ee[e];t===0||t===10?((e===1?v:y)(w(n,0)),n.length=0):n.push(t)}function Oe(e,t,n,r){for(var i=0,a=0;a<n;a++){var o=M[t>>>2],s=M[t+4>>>2];t+=8;for(var c=0;c<s;c++)De(e,A[o+c>>>0]);i+=s}return M[r>>>2]=i,0}function H(e){return n[`_`+e]}function ke(e,t){k.set(e,t>>>0)}function Ae(e,t,n,r,i){var a={string:e=>{var t=0;if(e!=null&&e!==0){var n=(e.length<<2)+1;t=fr(n),D(e,t,n)}return t},array:e=>{var t=fr(e.length);return ke(e,t),t}};function o(e){return t===`string`?T(e):t===`boolean`?!!e:e}var s=H(e),c=[],l=0;if(r)for(var u=0;u<r.length;u++){var d=a[n[u]];d?(l===0&&(l=ur()),c[u]=d(r[u])):c[u]=r[u]}var f=s.apply(null,c);function p(e){return l!==0&&dr(l),o(e)}return f=p(f),f}function je(e,t,n,r){n||=[];var i=n.every(e=>e===`number`||e===`boolean`);return t!==`string`&&i&&!r?H(e):function(){return Ae(e,t,n,arguments,r)}}var Me={abort:ve,emscripten_get_heap_max:be,emscripten_memcpy_big:V,emscripten_resize_heap:Se,fd_close:we,fd_seek:Te,fd_write:Oe};he();var Ne=n.___wasm_call_ctors=function(){return(Ne=n.___wasm_call_ctors=n.asm.__wasm_call_ctors).apply(null,arguments)},Pe=n._init=function(){return(Pe=n._init=n.asm.init).apply(null,arguments)},Fe=n._init_with_threads_count=function(){return(Fe=n._init_with_threads_count=n.asm.init_with_threads_count).apply(null,arguments)},Ie=n._get_threads_count=function(){return(Ie=n._get_threads_count=n.asm.get_threads_count).apply(null,arguments)},Le=n._register_tensor=function(){return(Le=n._register_tensor=n.asm.register_tensor).apply(null,arguments)},Re=n._dispose_data=function(){return(Re=n._dispose_data=n.asm.dispose_data).apply(null,arguments)},ze=n._dispose=function(){return(ze=n._dispose=n.asm.dispose).apply(null,arguments)},Be=n._Abs=function(){return(Be=n._Abs=n.asm.Abs).apply(null,arguments)},Ve=n._Acos=function(){return(Ve=n._Acos=n.asm.Acos).apply(null,arguments)},He=n._Acosh=function(){return(He=n._Acosh=n.asm.Acosh).apply(null,arguments)},U=n._Add=function(){return(U=n._Add=n.asm.Add).apply(null,arguments)},Ue=n._AddN=function(){return(Ue=n._AddN=n.asm.AddN).apply(null,arguments)},We=n._All=function(){return(We=n._All=n.asm.All).apply(null,arguments)},Ge=n._Any=function(){return(Ge=n._Any=n.asm.Any).apply(null,arguments)},Ke=n._ArgMax=function(){return(Ke=n._ArgMax=n.asm.ArgMax).apply(null,arguments)},qe=n._ArgMin=function(){return(qe=n._ArgMin=n.asm.ArgMin).apply(null,arguments)},Je=n._Asin=function(){return(Je=n._Asin=n.asm.Asin).apply(null,arguments)},W=n._Asinh=function(){return(W=n._Asinh=n.asm.Asinh).apply(null,arguments)},Ye=n._Atan=function(){return(Ye=n._Atan=n.asm.Atan).apply(null,arguments)},Xe=n._Atan2=function(){return(Xe=n._Atan2=n.asm.Atan2).apply(null,arguments)},Ze=n._Atanh=function(){return(Ze=n._Atanh=n.asm.Atanh).apply(null,arguments)},Qe=n._AvgPool=function(){return(Qe=n._AvgPool=n.asm.AvgPool).apply(null,arguments)},$e=n._AvgPool3D=function(){return($e=n._AvgPool3D=n.asm.AvgPool3D).apply(null,arguments)},et=n._AvgPool3DGrad=function(){return(et=n._AvgPool3DGrad=n.asm.AvgPool3DGrad).apply(null,arguments)},tt=n._AvgPoolGrad=function(){return(tt=n._AvgPoolGrad=n.asm.AvgPoolGrad).apply(null,arguments)},G=n._BatchMatMul=function(){return(G=n._BatchMatMul=n.asm.BatchMatMul).apply(null,arguments)},nt=n._Bincount=function(){return(nt=n._Bincount=n.asm.Bincount).apply(null,arguments)},rt=n._BitwiseAnd=function(){return(rt=n._BitwiseAnd=n.asm.BitwiseAnd).apply(null,arguments)},it=n._Ceil=function(){return(it=n._Ceil=n.asm.Ceil).apply(null,arguments)},at=n._ClipByValue=function(){return(at=n._ClipByValue=n.asm.ClipByValue).apply(null,arguments)},ot=n._Conv2D=function(){return(ot=n._Conv2D=n.asm.Conv2D).apply(null,arguments)},st=n._Conv2DBackpropInput=function(){return(st=n._Conv2DBackpropInput=n.asm.Conv2DBackpropInput).apply(null,arguments)},ct=n._Conv3D=function(){return(ct=n._Conv3D=n.asm.Conv3D).apply(null,arguments)},lt=n._Conv3DBackpropFilterV2=function(){return(lt=n._Conv3DBackpropFilterV2=n.asm.Conv3DBackpropFilterV2).apply(null,arguments)},ut=n._Conv3DBackpropInputV2=function(){return(ut=n._Conv3DBackpropInputV2=n.asm.Conv3DBackpropInputV2).apply(null,arguments)},dt=n._Cos=function(){return(dt=n._Cos=n.asm.Cos).apply(null,arguments)},ft=n._Cosh=function(){return(ft=n._Cosh=n.asm.Cosh).apply(null,arguments)},pt=n._CropAndResize=function(){return(pt=n._CropAndResize=n.asm.CropAndResize).apply(null,arguments)},mt=n._Cumprod=function(){return(mt=n._Cumprod=n.asm.Cumprod).apply(null,arguments)},ht=n._Cumsum=function(){return(ht=n._Cumsum=n.asm.Cumsum).apply(null,arguments)},gt=n._DenseBincount=function(){return(gt=n._DenseBincount=n.asm.DenseBincount).apply(null,arguments)},_t=n._DepthToSpace=function(){return(_t=n._DepthToSpace=n.asm.DepthToSpace).apply(null,arguments)},vt=n._DepthwiseConv2dNative=function(){return(vt=n._DepthwiseConv2dNative=n.asm.DepthwiseConv2dNative).apply(null,arguments)},yt=n._Diag=function(){return(yt=n._Diag=n.asm.Diag).apply(null,arguments)},bt=n._Dilation2D=function(){return(bt=n._Dilation2D=n.asm.Dilation2D).apply(null,arguments)},xt=n._Dilation2DBackpropFilter=function(){return(xt=n._Dilation2DBackpropFilter=n.asm.Dilation2DBackpropFilter).apply(null,arguments)},St=n._Dilation2DBackpropInput=function(){return(St=n._Dilation2DBackpropInput=n.asm.Dilation2DBackpropInput).apply(null,arguments)},Ct=n._Elu=function(){return(Ct=n._Elu=n.asm.Elu).apply(null,arguments)},wt=n._EluGrad=function(){return(wt=n._EluGrad=n.asm.EluGrad).apply(null,arguments)},Tt=n._Equal=function(){return(Tt=n._Equal=n.asm.Equal).apply(null,arguments)},Et=n._Erf=function(){return(Et=n._Erf=n.asm.Erf).apply(null,arguments)},Dt=n._Exp=function(){return(Dt=n._Exp=n.asm.Exp).apply(null,arguments)},Ot=n._Expm1=function(){return(Ot=n._Expm1=n.asm.Expm1).apply(null,arguments)},kt=n._FlipLeftRight=function(){return(kt=n._FlipLeftRight=n.asm.FlipLeftRight).apply(null,arguments)},At=n._Floor=function(){return(At=n._Floor=n.asm.Floor).apply(null,arguments)},jt=n._FloorDiv=function(){return(jt=n._FloorDiv=n.asm.FloorDiv).apply(null,arguments)},Mt=n._FusedBatchNorm=function(){return(Mt=n._FusedBatchNorm=n.asm.FusedBatchNorm).apply(null,arguments)},Nt=n._FusedConv2D=function(){return(Nt=n._FusedConv2D=n.asm.FusedConv2D).apply(null,arguments)},Pt=n._FusedDepthwiseConv2D=function(){return(Pt=n._FusedDepthwiseConv2D=n.asm.FusedDepthwiseConv2D).apply(null,arguments)},Ft=n._Gather=function(){return(Ft=n._Gather=n.asm.Gather).apply(null,arguments)},It=n._GatherNd=function(){return(It=n._GatherNd=n.asm.GatherNd).apply(null,arguments)},Lt=n._Greater=function(){return(Lt=n._Greater=n.asm.Greater).apply(null,arguments)},Rt=n._GreaterEqual=function(){return(Rt=n._GreaterEqual=n.asm.GreaterEqual).apply(null,arguments)},zt=n._IsFinite=function(){return(zt=n._IsFinite=n.asm.IsFinite).apply(null,arguments)},Bt=n._IsInf=function(){return(Bt=n._IsInf=n.asm.IsInf).apply(null,arguments)},Vt=n._IsNan=function(){return(Vt=n._IsNan=n.asm.IsNan).apply(null,arguments)},Ht=n._LRN=function(){return(Ht=n._LRN=n.asm.LRN).apply(null,arguments)},Ut=n._LRNGrad=function(){return(Ut=n._LRNGrad=n.asm.LRNGrad).apply(null,arguments)},Wt=n._LeakyRelu=function(){return(Wt=n._LeakyRelu=n.asm.LeakyRelu).apply(null,arguments)},Gt=n._Less=function(){return(Gt=n._Less=n.asm.Less).apply(null,arguments)},Kt=n._LessEqual=function(){return(Kt=n._LessEqual=n.asm.LessEqual).apply(null,arguments)},qt=n._LinSpace=function(){return(qt=n._LinSpace=n.asm.LinSpace).apply(null,arguments)},Jt=n._Log=function(){return(Jt=n._Log=n.asm.Log).apply(null,arguments)},Yt=n._Log1p=function(){return(Yt=n._Log1p=n.asm.Log1p).apply(null,arguments)},Xt=n._LogicalAnd=function(){return(Xt=n._LogicalAnd=n.asm.LogicalAnd).apply(null,arguments)},Zt=n._LogicalNot=function(){return(Zt=n._LogicalNot=n.asm.LogicalNot).apply(null,arguments)},Qt=n._LogicalOr=function(){return(Qt=n._LogicalOr=n.asm.LogicalOr).apply(null,arguments)},$t=n._LogicalXor=function(){return($t=n._LogicalXor=n.asm.LogicalXor).apply(null,arguments)},en=n._Max=function(){return(en=n._Max=n.asm.Max).apply(null,arguments)},tn=n._MaxPool=function(){return(tn=n._MaxPool=n.asm.MaxPool).apply(null,arguments)},nn=n._MaxPool3D=function(){return(nn=n._MaxPool3D=n.asm.MaxPool3D).apply(null,arguments)},rn=n._MaxPool3DGrad=function(){return(rn=n._MaxPool3DGrad=n.asm.MaxPool3DGrad).apply(null,arguments)},an=n._MaxPoolGrad=function(){return(an=n._MaxPoolGrad=n.asm.MaxPoolGrad).apply(null,arguments)},on=n._MaxPoolWithArgmax=function(){return(on=n._MaxPoolWithArgmax=n.asm.MaxPoolWithArgmax).apply(null,arguments)},sn=n._Maximum=function(){return(sn=n._Maximum=n.asm.Maximum).apply(null,arguments)},K=n._Mean=function(){return(K=n._Mean=n.asm.Mean).apply(null,arguments)},cn=n._Min=function(){return(cn=n._Min=n.asm.Min).apply(null,arguments)},ln=n._Minimum=function(){return(ln=n._Minimum=n.asm.Minimum).apply(null,arguments)},un=n._MirrorPad=function(){return(un=n._MirrorPad=n.asm.MirrorPad).apply(null,arguments)},dn=n._Mod=function(){return(dn=n._Mod=n.asm.Mod).apply(null,arguments)},fn=n._Multinomial=function(){return(fn=n._Multinomial=n.asm.Multinomial).apply(null,arguments)},pn=n._Multiply=function(){return(pn=n._Multiply=n.asm.Multiply).apply(null,arguments)},mn=n._Neg=function(){return(mn=n._Neg=n.asm.Neg).apply(null,arguments)},hn=n._NonMaxSuppressionV3=function(){return(hn=n._NonMaxSuppressionV3=n.asm.NonMaxSuppressionV3).apply(null,arguments)},gn=n._NonMaxSuppressionV4=function(){return(gn=n._NonMaxSuppressionV4=n.asm.NonMaxSuppressionV4).apply(null,arguments)},_n=n._NonMaxSuppressionV5=function(){return(_n=n._NonMaxSuppressionV5=n.asm.NonMaxSuppressionV5).apply(null,arguments)},vn=n._NotEqual=function(){return(vn=n._NotEqual=n.asm.NotEqual).apply(null,arguments)},yn=n._OneHot=function(){return(yn=n._OneHot=n.asm.OneHot).apply(null,arguments)},bn=n._PadV2=function(){return(bn=n._PadV2=n.asm.PadV2).apply(null,arguments)},xn=n._Pow=function(){return(xn=n._Pow=n.asm.Pow).apply(null,arguments)},Sn=n._Prelu=function(){return(Sn=n._Prelu=n.asm.Prelu).apply(null,arguments)},Cn=n._Prod=function(){return(Cn=n._Prod=n.asm.Prod).apply(null,arguments)},wn=n._RealDiv=function(){return(wn=n._RealDiv=n.asm.RealDiv).apply(null,arguments)},Tn=n._Reciprocal=function(){return(Tn=n._Reciprocal=n.asm.Reciprocal).apply(null,arguments)},q=n._Relu=function(){return(q=n._Relu=n.asm.Relu).apply(null,arguments)},En=n._Relu6=function(){return(En=n._Relu6=n.asm.Relu6).apply(null,arguments)},Dn=n._ResizeBilinear=function(){return(Dn=n._ResizeBilinear=n.asm.ResizeBilinear).apply(null,arguments)},On=n._ResizeBilinearGrad=function(){return(On=n._ResizeBilinearGrad=n.asm.ResizeBilinearGrad).apply(null,arguments)},kn=n._ResizeNearestNeighbor=function(){return(kn=n._ResizeNearestNeighbor=n.asm.ResizeNearestNeighbor).apply(null,arguments)},An=n._ResizeNearestNeighborGrad=function(){return(An=n._ResizeNearestNeighborGrad=n.asm.ResizeNearestNeighborGrad).apply(null,arguments)},jn=n._Reverse=function(){return(jn=n._Reverse=n.asm.Reverse).apply(null,arguments)},Mn=n._RotateWithOffset=function(){return(Mn=n._RotateWithOffset=n.asm.RotateWithOffset).apply(null,arguments)},Nn=n._Round=function(){return(Nn=n._Round=n.asm.Round).apply(null,arguments)},Pn=n._Rsqrt=function(){return(Pn=n._Rsqrt=n.asm.Rsqrt).apply(null,arguments)},Fn=n._ScatterNd=function(){return(Fn=n._ScatterNd=n.asm.ScatterNd).apply(null,arguments)},In=n._SearchSorted=function(){return(In=n._SearchSorted=n.asm.SearchSorted).apply(null,arguments)},Ln=n._SelectV2=function(){return(Ln=n._SelectV2=n.asm.SelectV2).apply(null,arguments)},Rn=n._Selu=function(){return(Rn=n._Selu=n.asm.Selu).apply(null,arguments)},zn=n._Sigmoid=function(){return(zn=n._Sigmoid=n.asm.Sigmoid).apply(null,arguments)},Bn=n._Sign=function(){return(Bn=n._Sign=n.asm.Sign).apply(null,arguments)},Vn=n._Sin=function(){return(Vn=n._Sin=n.asm.Sin).apply(null,arguments)},Hn=n._Sinh=function(){return(Hn=n._Sinh=n.asm.Sinh).apply(null,arguments)},Un=n._Softmax=function(){return(Un=n._Softmax=n.asm.Softmax).apply(null,arguments)},Wn=n._Softplus=function(){return(Wn=n._Softplus=n.asm.Softplus).apply(null,arguments)},Gn=n._SparseFillEmptyRows=function(){return(Gn=n._SparseFillEmptyRows=n.asm.SparseFillEmptyRows).apply(null,arguments)},Kn=n._SparseReshape=function(){return(Kn=n._SparseReshape=n.asm.SparseReshape).apply(null,arguments)},qn=n._SparseSegmentReduction=function(){return(qn=n._SparseSegmentReduction=n.asm.SparseSegmentReduction).apply(null,arguments)},Jn=n._SparseToDense=function(){return(Jn=n._SparseToDense=n.asm.SparseToDense).apply(null,arguments)},Yn=n._Sqrt=function(){return(Yn=n._Sqrt=n.asm.Sqrt).apply(null,arguments)},Xn=n._Square=function(){return(Xn=n._Square=n.asm.Square).apply(null,arguments)},Zn=n._SquaredDifference=function(){return(Zn=n._SquaredDifference=n.asm.SquaredDifference).apply(null,arguments)},Qn=n._Step=function(){return(Qn=n._Step=n.asm.Step).apply(null,arguments)},$n=n._StridedSlice=function(){return($n=n._StridedSlice=n.asm.StridedSlice).apply(null,arguments)},er=n._Sub=function(){return(er=n._Sub=n.asm.Sub).apply(null,arguments)},J=n._Sum=function(){return(J=n._Sum=n.asm.Sum).apply(null,arguments)},tr=n._Tan=function(){return(tr=n._Tan=n.asm.Tan).apply(null,arguments)},nr=n._Tanh=function(){return(nr=n._Tanh=n.asm.Tanh).apply(null,arguments)},rr=n._TensorScatterUpdate=function(){return(rr=n._TensorScatterUpdate=n.asm.TensorScatterUpdate).apply(null,arguments)},ir=n._Tile=function(){return(ir=n._Tile=n.asm.Tile).apply(null,arguments)},ar=n._TopK=function(){return(ar=n._TopK=n.asm.TopK).apply(null,arguments)},Y=n._Transform=function(){return(Y=n._Transform=n.asm.Transform).apply(null,arguments)},or=n._Transpose=function(){return(or=n._Transpose=n.asm.Transpose).apply(null,arguments)},sr=n.__FusedMatMul=function(){return(sr=n.__FusedMatMul=n.asm._FusedMatMul).apply(null,arguments)},cr=n._malloc=function(){return(cr=n._malloc=n.asm.malloc).apply(null,arguments)},X=n._free=function(){return(X=n._free=n.asm.free).apply(null,arguments)},lr=n.___errno_location=function(){return(lr=n.___errno_location=n.asm.__errno_location).apply(null,arguments)},ur=n.stackSave=function(){return(ur=n.stackSave=n.asm.stackSave).apply(null,arguments)},dr=n.stackRestore=function(){return(dr=n.stackRestore=n.asm.stackRestore).apply(null,arguments)},fr=n.stackAlloc=function(){return(fr=n.stackAlloc=n.asm.stackAlloc).apply(null,arguments)},pr=n.dynCall_iijjiiii=function(){return(pr=n.dynCall_iijjiiii=n.asm.dynCall_iijjiiii).apply(null,arguments)},mr=n.dynCall_jiji=function(){return(mr=n.dynCall_jiji=n.asm.dynCall_jiji).apply(null,arguments)};n.cwrap=je;var hr;R=function e(){hr||gr(),hr||(R=e)};function gr(e){if(e||=s,L>0||(te(),L>0))return;function t(){hr||(hr=!0,n.calledRun=!0,!S&&(ne(),r(n),n.onRuntimeInitialized&&n.onRuntimeInitialized(),ie()))}n.setStatus?(n.setStatus(`Running...`),setTimeout(function(){setTimeout(function(){n.setStatus(``)},1),t()},1)):t()}if(n.preInit)for(typeof n.preInit==`function`&&(n.preInit=[n.preInit]);n.preInit.length>0;)n.preInit.pop()();gr();var _r;a&&(_r={uncaughtException:process.listeners(`uncaughtException`).filter(function(e){return!a.uncaughtException.indexOf(e)>-1}),unhandledRejection:process.listeners(`unhandledRejection`).filter(function(e){return!a.unhandledRejection.indexOf(e)>-1})});var Z;if(t!==void 0)Z=t;else if(typeof WasmBackendModuleThreadedSimd<`u`)Z=WasmBackendModuleThreadedSimd;else throw Error(`Could not find wasm module in post.js`);if(_r){var vr=Z._dispose;Z._dispose=function(){vr(),_r.uncaughtException.forEach(function(e){process.removeListener(`uncaughtException`,e)}),_r.unhandledRejection.forEach(function(e){process.removeListener(`unhandledRejection`,e)})}}return t.ready})})();typeof e==`object`&&typeof t==`object`?t.exports=n:typeof define==`function`&&define.amd?define([],function(){return n}):typeof e==`object`&&(e.WasmBackendModule=n)})),$u=/* @__PURE__ */ S(Xu()),ed=Zu(),td=/* @__PURE__ */ S(Qu());
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
const nd=$u.default||$u,rd=td.default||td;var id=class extends f{constructor(e){super(),this.wasm=e,this.dataIdNextNumber=1,this.wasm.tfjs.initWithThreadsCount(_d),vd=this.wasm.tfjs.getThreadsCount(),this.dataIdMap=new Te(this,ve())}write(e,t,n){let r={id:this.dataIdNextNumber++};return this.move(r,e,t,n,1),r}numDataIds(){return this.dataIdMap.numDataIds()}async time(e){let t=Tn();return e(),{kernelMs:Tn()-t}}move(e,t,n,r,i){let a=this.dataIdNextNumber++;if(r===`string`){let o=t;this.dataIdMap.set(e,{id:a,stringBytes:o,shape:n,dtype:r,memoryOffset:null,refCount:i});return}let o=G(n),s=o*St(r),c=this.wasm._malloc(s)>>>0;this.dataIdMap.set(e,{id:a,memoryOffset:c,shape:n,dtype:r,refCount:i}),this.wasm.tfjs.registerTensor(a,o,c),t!=null&&this.wasm.HEAPU8.set(new Uint8Array(t.buffer,t.byteOffset,s),c)}async read(e){return this.readSync(e)}readSync(e,t,n){let{memoryOffset:r,dtype:i,shape:a,stringBytes:o}=this.dataIdMap.get(e);if(i===`string`)return(t==null||t===0)&&(n==null||n>=o.length)?o:o.slice(t,n);t||=0,n||=G(a);let s=St(i);return cd(this.wasm.HEAPU8.slice(r+t*s,r+n*s).buffer,i)}disposeData(e,t=!1){if(this.dataIdMap.has(e)){let n=this.dataIdMap.get(e);if(n.refCount--,!t&&n.refCount>0)return!1;this.wasm._free(n.memoryOffset),this.wasm.tfjs.disposeData(n.id),this.dataIdMap.delete(e)}return!0}refCount(e){return this.dataIdMap.has(e)?this.dataIdMap.get(e).refCount:0}incRef(e){let t=this.dataIdMap.get(e);t!=null&&t.refCount++}floatPrecision(){return 32}getMemoryOffset(e){return this.dataIdMap.get(e).memoryOffset}dispose(){this.wasm.tfjs.dispose(),`PThread`in this.wasm&&this.wasm.PThread.terminateAllThreads(),this.wasm=null}memory(){return{unreliable:!1}}makeOutput(e,t,n,r){let i;if(n==null)i=this.write(r??null,e,t);else{let r=this.dataIdNextNumber++;i={id:r},this.dataIdMap.set(i,{id:r,memoryOffset:n,shape:e,dtype:t,refCount:1});let a=G(e);this.wasm.tfjs.registerTensor(r,a,n)}return{dataId:i,shape:e,dtype:t}}typedArrayFromHeap({shape:e,dtype:t,dataId:n}){let r=this.wasm.HEAPU8.buffer,{memoryOffset:i}=this.dataIdMap.get(n),a=G(e);switch(t){case`float32`:return new Float32Array(r,i,a);case`int32`:return new Int32Array(r,i,a);case`bool`:return new Uint8Array(r,i,a);default:throw Error(`Unknown dtype ${t}`)}}};function ad(e){return(t,n)=>(Gt(e,{credentials:`same-origin`}).then(r=>{r.ok||t.env.a(`failed to load wasm binary file at '${e}'`),r.arrayBuffer().then(e=>{WebAssembly.instantiate(e,t).then(e=>{n(e.instance,e.module)})})}),{})}function od(e,t,n){if(ud!=null)return ud;let r=`tfjs-backend-wasm.wasm`;return e&&t?r=`tfjs-backend-wasm-threaded-simd.wasm`:e&&(r=`tfjs-backend-wasm-simd.wasm`),fd!=null&&fd[r]!=null?fd[r]:n+r}async function sd(){let[e,t]=await Promise.all([ct().getAsync(`WASM_HAS_SIMD_SUPPORT`),ct().getAsync(`WASM_HAS_MULTITHREAD_SUPPORT`)]);return new Promise((n,r)=>{let i={};i.locateFile=(n,r)=>{if(n.endsWith(`.worker.js`)){let e=ed.wasmWorkerContents.replace(/\n/g,`\\n`),t=new Blob([e],{type:`application/javascript`});return URL.createObjectURL(t)}return n.endsWith(`.wasm`)?od(e,t,dd??r):r+n},md&&(i.instantiateWasm=ad(od(e,t,dd??``)));let a=!1;i.onAbort=()=>{a||pd||(pd=!0,r({message:"Make sure the server can serve the `.wasm` file relative to the bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers"}))};let o;t&&e&&ud==null?(i.mainScriptUrlOrBlob=new Blob([`var WasmBackendModuleThreadedSimd = `+nd.toString()],{type:`text/javascript`}),o=nd(i)):o=rd(i),o.then(e=>{a=!0,pd=!1,e.tfjs={init:e.cwrap(`init`,null,[]),initWithThreadsCount:e.cwrap(`init_with_threads_count`,null,[`number`]),getThreadsCount:e.cwrap(`get_threads_count`,`number`,[]),registerTensor:e.cwrap(`register_tensor`,null,[`number`,`number`,`number`]),disposeData:e.cwrap(`dispose_data`,null,[`number`]),dispose:e.cwrap(`dispose`,null,[])},n({wasm:e})}).catch(r)})}function cd(e,t){switch(t){case`float32`:return new Float32Array(e);case`int32`:return new Int32Array(e);case`bool`:return new Uint8Array(e);default:throw Error(`Unknown dtype ${t}`)}}const ld=[`tfjs-backend-wasm.wasm`,`tfjs-backend-wasm-simd.wasm`,`tfjs-backend-wasm-threaded-simd.wasm`];let ud=null,dd=null,fd={},pd=!1,md=!1;function hd(e,t=!1){if(He(`setWasmPath has been deprecated in favor of setWasmPaths and will be removed in a future release.`),pd)throw Error("The WASM backend was already initialized. Make sure you call `setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`");ud=e,md=t}function gd(e,t=!1){if(pd)throw Error("The WASM backend was already initialized. Make sure you call `setWasmPaths()` before you call `tf.setBackend()` or `tf.ready()`");if(typeof e==`string`)dd=e;else{fd=e;let t=ld.filter(e=>fd[e]==null);if(t.length>0)throw Error(`There were no entries found for the following binaries: ${t.join(`,`)}. Please either call setWasmPaths with a map providing a path for each binary, or with a string indicating the directory where all the binaries can be found.`)}md=t}let _d=-1,vd=-1;function yd(e){_d=e}function bd(){if(vd===-1)throw Error(`WASM backend not initialized.`);return vd}
/** @license See the LICENSE file. */
const xd=`4.22.0`;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
gt(`wasm`,async()=>{let{wasm:e}=await sd();return new id(e)},2);
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
export{id as BackendWasm,bd as getThreadsCount,yd as setThreadsCount,hd as setWasmPath,gd as setWasmPaths,xd as version_wasm};
var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,n)=>{let r={};for(var i in e)t(r,i,{get:e[i],enumerable:!0});return n||t(r,Symbol.toStringTag,{value:`Module`}),r},c=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},l=(n,r,o)=>(o=n==null?{}:e(i(n)),c(r||!n||!n.__esModule||!a.call(n,`default`)?t(o,`default`,{value:n,enumerable:!0}):o,n)),u=class{constructor(e,t){
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
this.backend=e,this.dataMover=t,this.data=/* @__PURE__ */ new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}},d=class{refCount(e){return f(`refCount`)}incRef(e){return f(`incRef`)}timerAvailable(){return!0}time(e){return f(`time`)}read(e){return f(`read`)}readSync(e){return f(`readSync`)}readToGPU(e,t){return f(`readToGPU`)}numDataIds(){return f(`numDataIds`)}disposeData(e,t){return f(`disposeData`)}write(e,t,n){return f(`write`)}move(e,t,n,r,i){return f(`move`)}createTensorFromGPUData(e,t,n){return f(`createTensorFromGPUData`)}memory(){return f(`memory`)}floatPrecision(){return f(`floatPrecision`)}epsilon(){return this.floatPrecision()===32?1e-7:1e-4}dispose(){return f(`dispose`)}};function f(e){throw Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}
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
function p(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,_(e,t,n)}function m(e,t){if(e.length!==t.length)throw Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,r=0;for(;n>0;)r=Math.random()*n|0,n--,_(e,n,r),_(t,n,r)}function h(e,t,n){return Math.max(e,Math.min(t,n))}function g(e){return e%2==0?e:e+1}function _(e,t,n){let r=e[t];e[t]=e[n],e[n]=r}function v(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function y(e,t){let n=Math.random();return t*n+(1-n)*e}function b(e,t){let n=0;for(let r=0;r<e.length;r++){let i=Number(e[r])-Number(t[r]);n+=i*i}return n}function x(e,t){if(!e)throw Error(typeof t==`string`?t:t())}function S(e,t,n=``){x(E(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function C(e){x(e!=null,()=>`The input to the tensor constructor must be a non-null value.`)}function w(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function ee(e){return e.length===0}function T(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function E(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function te(e){return e%1==0}function ne(e){if(Math.tanh!=null)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{let t=Math.exp(2*e);return(t-1)/(t+1)}}function re(e){let t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function ie(e){let t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return p(t),t}function ae(e,t){return t<=e.length?e:e+` `.repeat(t-e.length)}function oe(e,t=e=>0,n,r){return new Promise((i,a)=>{let o=0,s=()=>{if(e()){i();return}o++;let c=t(o);if(n!=null&&o>=n){a();return}r==null?setTimeout(s,c):r(s,c)};s()})}function se(e,t){let n=1,r=-1;for(let t=0;t<e.length;++t)if(e[t]>=0)n*=e[t];else if(e[t]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${t}`);r=t}else if(e[t]<0)throw Error(`Shapes can not be < 0. Found ${e[t]} at dim ${t}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);let i=e.slice();return i[r]=t/n,i}function ce(e,t){let n=t.length;return e=e==null?t.map((e,t)=>t):[].concat(e),x(e.every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),x(e.every(e=>te(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function le(e,t){let n=[],r=[],i=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||i?null:ce(t,e).sort(),o=0;for(let t=0;t<e.length;++t){if(a!=null){if(a[o]===t&&e[t]!==1)throw Error(`Can't squeeze axis ${t} since its dim '${e[t]}' is not 1`);(a[o]==null||a[o]>t)&&e[t]===1&&(n.push(e[t]),r.push(t)),a[o]<=t&&o++}e[t]!==1&&(n.push(e[t]),r.push(t))}return{newShape:n,keptDims:r}}function ue(e,t){return de(e,t)}function de(e,t){let n=null;if(e==null||e===`float32`)n=new Float32Array(t);else if(e===`int32`)n=new Int32Array(t);else if(e===`bool`)n=new Uint8Array(t);else if(e===`string`)n=Array(t);else throw Error(`Unknown data type ${e}`);return n}function fe(e,t){for(let n=0;n<e.length;n++){let r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function pe(e){return e===`bool`||e===`complex64`||e===`float32`||e===`int32`||e===`string`}function me(e,t){return!(t===`complex64`||t===`float32`&&e!==`complex64`||t===`int32`&&e!==`float32`&&e!==`complex64`||t===`bool`&&e===`bool`)}function he(e){if(e===`float32`||e===`int32`)return 4;if(e===`complex64`)return 8;if(e===`bool`)return 1;throw Error(`Unknown dtype ${e}`)}function ge(e){if(e==null)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function _e(e){return typeof e==`string`||e instanceof String}function ve(e){return typeof e==`boolean`}function ye(e){return typeof e==`number`}function be(e){return Array.isArray(e)?be(e[0]):e instanceof Float32Array?`float32`:e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?`int32`:ye(e)?`float32`:_e(e)?`string`:ve(e)?`bool`:`float32`}function xe(e){return!!(e&&e.constructor&&e.call&&e.apply)}function Se(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function Ce(e){let t=e.length;if(t<2)return[];let n=Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function we(e,t,n,r=!1){let i=[];if(t.length===1){let a=t[0]*(r?2:1);for(let t=0;t<a;t++)i[t]=n[e+t]}else{let a=t[0],o=t.slice(1),s=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<a;t++)i[t]=we(e+t*s,o,n,r)}return i}function Te(e,t,n=!1){if(e.length===0)return t[0];let r=e.reduce((e,t)=>e*t)*(n?2:1);if(r===0)return[];if(r!==t.length)throw Error(`[${e}] does not match the input size ${t.length}${n?` for a complex tensor`:``}.`);return we(0,e,t,n)}function Ee(e,t){if(Array.isArray(e))return e;if(t===`float32`)return e instanceof Float32Array?e:new Float32Array(e);if(t===`int32`)return e instanceof Int32Array?e:new Int32Array(e);if(t===`bool`||t===`string`)return Uint8Array.from(new Int32Array(e));throw Error(`Unknown dtype ${t}`)}function De(e,t){let n=Oe(e,t);for(let e=0;e<n.length;e++)n[e]=1;return n}function Oe(e,t){if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`)return new Uint8Array(e);throw Error(`Unknown data type ${t}`)}function ke(e,t){let n=e.reduce((e,t)=>e*t,1);if(t==null||t===`float32`)return Te(e,new Float32Array(n));if(t===`int32`)return Te(e,new Int32Array(n));if(t===`bool`)return Te(e,new Uint8Array(n));throw Error(`Unknown data type ${t}`)}function Ae(e){e.forEach(t=>{x(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function je(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let t=0;t<e.length-1;++t)r+=n[t]*e[t];return r}function Me(e,t,n){if(t===0)return[];if(t===1)return[e];let r=Array(t);for(let t=0;t<r.length-1;++t)r[t]=Math.floor(e/n[t]),e-=r[t]*n[t];return r[r.length-1]=e,r}function Ne(e){return e&&e.then&&typeof e.then==`function`}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
const Pe=`tfjsflags`;var Fe=class{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Ie,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(D().getBool(`IS_TEST`)||D().getBool(`PROD`)||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){let t=this.urlFlags[e];D().getBool(`IS_TEST`)||D().getBool(`PROD`)||console.warn(`Setting feature override from URL ${e}: ${t}.`),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];let t=this.evaluateFlag(e);if(Ne(t))throw Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(this.global===void 0||this.global.location===void 0||this.global.location.search===void 0)return;let e=this.getQueryParams(this.global.location.search);Pe in e&&e[Pe].split(`,`).forEach(e=>{let[t,n]=e.split(`:`);this.urlFlags[t]=Re(t,n)})}};function Ie(e){let t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(Le(t,n[0],n[1]),n.join(`=`))),t}function Le(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||``)}function Re(e,t){let n=t.toLowerCase();return n===`true`||n===`false`?n===`true`:`${+n}`===n?+n:t}function D(){return ze}let ze=null;function Be(e){ze=e}
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
let Ve;function He(){if(Ve==null){let e;if(typeof window<`u`)e=window;else if(typeof global<`u`)e=global;else if(typeof process<`u`)e=process;else if(typeof self<`u`)e=self;else throw Error(`Could not find a global object`);Ve=e}return Ve}function Ue(){let e=He();return e._tfGlobals??=/* @__PURE__ */ new Map,e._tfGlobals}function We(e,t){let n=Ue();if(n.has(e))return n.get(e);{let r=t();return n.set(e,r),n.get(e)}}const Ge=`Abs`,Ke=`Acos`,qe=`Acosh`,Je=`Add`,Ye=`AddN`,Xe=`All`,Ze=`Any`,Qe=`ArgMax`,$e=`ArgMin`,et=`Asin`,tt=`Asinh`,nt=`Atan`,rt=`Atanh`,it=`Atan2`,at=`AvgPool`,ot=`AvgPoolGrad`,st=`AvgPool3D`,ct=`AvgPool3DGrad`,lt=`BatchMatMul`,ut=`BatchToSpaceND`,dt=`Bincount`,ft=`BitwiseAnd`,pt=`BroadcastArgs`,mt=`Cast`,ht=`Ceil`,gt=`ClipByValue`,_t=`Complex`,vt=`ComplexAbs`,yt=`Concat`,bt=`Conv2D`,xt=`Conv2DBackpropFilter`,St=`Conv2DBackpropInput`,Ct=`Conv3D`,wt=`Conv3DBackpropFilterV2`,Tt=`Conv3DBackpropInputV2`,Et=`Cos`,Dt=`Cosh`,Ot=`Cumprod`,kt=`Cumsum`,At=`CropAndResize`,jt=`DenseBincount`,Mt=`DepthToSpace`,Nt=`DepthwiseConv2dNative`,Pt=`DepthwiseConv2dNativeBackpropFilter`,Ft=`DepthwiseConv2dNativeBackpropInput`,It=`Diag`,Lt=`Dilation2D`,Rt=`Dilation2DBackpropInput`,zt=`Dilation2DBackpropFilter`,Bt=`Draw`,Vt=`RealDiv`,Ht=`Einsum`,Ut=`Elu`,Wt=`EluGrad`,Gt=`Erf`,Kt=`Equal`,qt=`Exp`,Jt=`ExpandDims`,Yt=`Expm1`,Xt=`Fill`,Zt=`FlipLeftRight`,Qt=`Floor`,$t=`FloorDiv`,en=`FusedBatchNorm`,tn=`GatherV2`,nn=`GatherNd`,rn=`Greater`,an=`GreaterEqual`,on=`Identity`,sn=`IFFT`,cn=`Imag`,ln=`IsFinite`,un=`IsInf`,dn=`IsNan`,fn=`LeakyRelu`,pn=`Less`,mn=`LessEqual`,hn=`LinSpace`,gn=`Log`,_n=`Log1p`,vn=`LogicalAnd`,yn=`LogicalNot`,bn=`LogicalOr`,xn=`LogicalXor`,Sn=`LRN`,Cn=`LRNGrad`,wn=`Max`,Tn=`Maximum`,En=`MaxPool`,Dn=`MaxPoolGrad`,On=`MaxPool3D`,kn=`MaxPool3DGrad`,An=`MaxPoolWithArgmax`,jn=`Mean`,Mn=`Min`,Nn=`Minimum`,Pn=`MirrorPad`,Fn=`Mod`,In=`Multinomial`,Ln=`Multiply`,Rn=`Neg`,zn=`NotEqual`,Bn=`NonMaxSuppressionV3`,Vn=`NonMaxSuppressionV4`,Hn=`NonMaxSuppressionV5`,Un=`OnesLike`,Wn=`OneHot`,Gn=`Pack`,Kn=`PadV2`,qn=`Pow`,Jn=`Prelu`,Yn=`Prod`,Xn=`RaggedGather`,Zn=`RaggedRange`,Qn=`RaggedTensorToTensor`,$n=`Range`,er=`Real`,tr=`Reciprocal`,nr=`Relu`,rr=`Reshape`,ir=`ResizeNearestNeighbor`,ar=`ResizeNearestNeighborGrad`,or=`ResizeBilinear`,sr=`ResizeBilinearGrad`,cr=`Relu6`,lr=`Reverse`,ur=`Round`,dr=`Rsqrt`,fr=`ScatterNd`,pr=`TensorScatterUpdate`,mr=`SearchSorted`,hr=`Select`,gr=`Selu`,_r=`Slice`,vr=`Sin`,yr=`Sinh`,br=`Sign`,xr=`Sigmoid`,Sr=`Softplus`,Cr=`Sqrt`,wr=`Sum`,Tr=`SpaceToBatchND`,Er=`SplitV`,Dr=`Softmax`,Or=`SparseFillEmptyRows`,kr=`SparseReshape`,Ar=`SparseSegmentMean`,jr=`SparseSegmentSum`,Mr=`SparseToDense`,Nr=`SquaredDifference`,Pr=`Square`,Fr=`StaticRegexReplace`,Ir=`StridedSlice`,Lr=`StringNGrams`,Rr=`StringSplit`,zr=`StringToHashBucketFast`,Br=`Sub`,Vr=`Tan`,Hr=`Tanh`,Ur=`Tile`,Wr=`TopK`,Gr=`Transform`,Kr=`Transpose`,qr=`Unique`,Jr=`Unpack`,Yr=`UnsortedSegmentSum`,Xr=`ZerosLike`,Zr=`Step`,Qr=`FromPixels`,$r=`RotateWithOffset`,ei=`_FusedMatMul`,ti=`FusedConv2D`,ni=`FusedDepthwiseConv2D`;
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function ri(...e){D().getBool(`IS_TEST`)||D().getBool(`PROD`)||console.warn(...e)}function ii(...e){D().getBool(`IS_TEST`)||D().getBool(`PROD`)||console.log(...e)}
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
const ai=We(`kernelRegistry`,()=>/* @__PURE__ */ new Map),oi=We(`gradRegistry`,()=>/* @__PURE__ */ new Map);function si(e,t){let n=hi(e,t);return ai.get(n)}function ci(e){return oi.get(e)}function li(e){let t=ai.entries(),n=[];for(;;){let{done:r,value:i}=t.next();if(r)break;let[a,o]=i,[s]=a.split(`_`);s===e&&n.push(o)}return n}function ui(e){let{kernelName:t,backendName:n}=e,r=hi(t,n);ai.has(r)&&ri(`The kernel '${t}' for backend '${n}' is already registered`),ai.set(r,e)}function di(e){let{kernelName:t}=e;oi.has(t)&&D().getBool(`DEBUG`)&&ri(`Overriding the gradient for '${t}'`),oi.set(t,e)}function fi(e,t){let n=hi(e,t);if(!ai.has(n))throw Error(`The kernel '${e}' for backend '${t}' is not registered`);ai.delete(n)}function pi(e){if(!oi.has(e))throw Error(`The gradient '${e}' for backend is not registered`);oi.delete(e)}function mi(e,t){li(e).forEach(e=>{ui(Object.assign({},e,{backendName:t}))})}function hi(e,t){return`${t}_${e}`}
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
function gi(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var _i=/* @__PURE__ */ l((/* @__PURE__ */ o(((e,t)=>{t.exports=r;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function r(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}r.prototype.__isLong__,Object.defineProperty(r.prototype,"__isLong__",{value:!0});function i(e){return(e&&e.__isLong__)===!0}r.isLong=i;var a={},o={};function s(e,t){var n,r,i;return t?(e>>>=0,(i=0<=e&&e<256)&&(r=o[e],r)?r:(n=l(e,(e|0)<0?-1:0,!0),i&&(o[e]=n),n)):(e|=0,(i=-128<=e&&e<128)&&(r=a[e],r)?r:(n=l(e,e<0?-1:0,!1),i&&(a[e]=n),n))}r.fromInt=s;function c(e,t){if(isNaN(e))return t?b:y;if(t){if(e<0)return b;if(e>=g)return ee}else{if(e<=-_)return T;if(e+1>=_)return w}return e<0?c(-e,t).neg():l(e%h|0,e/h|0,t)}r.fromNumber=c;function l(e,t,n){return new r(e,t,n)}r.fromBits=l;var u=Math.pow;function d(e,t,n){if(e.length===0)throw Error(`empty string`);if(e===`NaN`||e===`Infinity`||e===`+Infinity`||e===`-Infinity`)return y;if(typeof t==`number`?(n=t,t=!1):t=!!t,n||=10,n<2||36<n)throw RangeError(`radix`);var r;if((r=e.indexOf(`-`))>0)throw Error(`interior hyphen`);if(r===0)return d(e.substring(1),t,n).neg();for(var i=c(u(n,8)),a=y,o=0;o<e.length;o+=8){var s=Math.min(8,e.length-o),l=parseInt(e.substring(o,o+s),n);if(s<8){var f=c(u(n,s));a=a.mul(f).add(c(l))}else a=a.mul(i),a=a.add(c(l))}return a.unsigned=t,a}r.fromString=d;function f(e,t){return typeof e==`number`?c(e,t):typeof e==`string`?d(e,t):l(e.low,e.high,typeof t==`boolean`?t:e.unsigned)}r.fromValue=f;var p=65536,m=1<<24,h=p*p,g=h*h,_=g/2,v=s(m),y=s(0);r.ZERO=y;var b=s(0,!0);r.UZERO=b;var x=s(1);r.ONE=x;var S=s(1,!0);r.UONE=S;var C=s(-1);r.NEG_ONE=C;var w=l(-1,2147483647,!1);r.MAX_VALUE=w;var ee=l(-1,-1,!0);r.MAX_UNSIGNED_VALUE=ee;var T=l(0,-2147483648,!1);r.MIN_VALUE=T;var E=r.prototype;E.toInt=function(){return this.unsigned?this.low>>>0:this.low},E.toNumber=function(){return this.unsigned?(this.high>>>0)*h+(this.low>>>0):this.high*h+(this.low>>>0)},E.toString=function(e){if(e||=10,e<2||36<e)throw RangeError(`radix`);if(this.isZero())return`0`;if(this.isNegative()){if(this.eq(T)){var t=c(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}return`-`+this.neg().toString(e)}for(var i=c(u(e,6),this.unsigned),a=this,o=``;;){var s=a.div(i),l=(a.sub(s.mul(i)).toInt()>>>0).toString(e);if(a=s,a.isZero())return l+o;for(;l.length<6;)l=`0`+l;o=``+l+o}},E.getHighBits=function(){return this.high},E.getHighBitsUnsigned=function(){return this.high>>>0},E.getLowBits=function(){return this.low},E.getLowBitsUnsigned=function(){return this.low>>>0},E.getNumBitsAbs=function(){if(this.isNegative())return this.eq(T)?64:this.neg().getNumBitsAbs();for(var e=this.high==0?this.low:this.high,t=31;t>0&&!(e&1<<t);t--);return this.high==0?t+1:t+33},E.isZero=function(){return this.high===0&&this.low===0},E.eqz=E.isZero,E.isNegative=function(){return!this.unsigned&&this.high<0},E.isPositive=function(){return this.unsigned||this.high>=0},E.isOdd=function(){return(this.low&1)==1},E.isEven=function(){return!(this.low&1)},E.equals=function(e){return i(e)||(e=f(e)),this.unsigned!==e.unsigned&&this.high>>>31==1&&e.high>>>31==1?!1:this.high===e.high&&this.low===e.low},E.eq=E.equals,E.notEquals=function(e){return!this.eq(e)},E.neq=E.notEquals,E.ne=E.notEquals,E.lessThan=function(e){return this.comp(e)<0},E.lt=E.lessThan,E.lessThanOrEqual=function(e){return this.comp(e)<=0},E.lte=E.lessThanOrEqual,E.le=E.lessThanOrEqual,E.greaterThan=function(e){return this.comp(e)>0},E.gt=E.greaterThan,E.greaterThanOrEqual=function(e){return this.comp(e)>=0},E.gte=E.greaterThanOrEqual,E.ge=E.greaterThanOrEqual,E.compare=function(e){if(i(e)||(e=f(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},E.comp=E.compare,E.negate=function(){return!this.unsigned&&this.eq(T)?T:this.not().add(x)},E.neg=E.negate,E.add=function(e){i(e)||(e=f(e));var t=this.high>>>16,n=this.high&65535,r=this.low>>>16,a=this.low&65535,o=e.high>>>16,s=e.high&65535,c=e.low>>>16,u=e.low&65535,d=0,p=0,m=0,h=0;return h+=a+u,m+=h>>>16,h&=65535,m+=r+c,p+=m>>>16,m&=65535,p+=n+s,d+=p>>>16,p&=65535,d+=t+o,d&=65535,l(m<<16|h,d<<16|p,this.unsigned)},E.subtract=function(e){return i(e)||(e=f(e)),this.add(e.neg())},E.sub=E.subtract,E.multiply=function(e){if(this.isZero())return y;if(i(e)||(e=f(e)),n)return l(n.mul(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(e.isZero())return y;if(this.eq(T))return e.isOdd()?T:y;if(e.eq(T))return this.isOdd()?T:y;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(v)&&e.lt(v))return c(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,s=e.high>>>16,u=e.high&65535,d=e.low>>>16,p=e.low&65535,m=0,h=0,g=0,_=0;return _+=o*p,g+=_>>>16,_&=65535,g+=a*p,h+=g>>>16,g&=65535,g+=o*d,h+=g>>>16,g&=65535,h+=r*p,m+=h>>>16,h&=65535,h+=a*d,m+=h>>>16,h&=65535,h+=o*u,m+=h>>>16,h&=65535,m+=t*p+r*d+a*u+o*s,m&=65535,l(g<<16|_,m<<16|h,this.unsigned)},E.mul=E.multiply,E.divide=function(e){if(i(e)||(e=f(e)),e.isZero())throw Error(`division by zero`);if(n)return!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1?this:l((this.unsigned?n.div_u:n.div_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(this.isZero())return this.unsigned?b:y;var t,r,a;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return b;if(e.gt(this.shru(1)))return S;a=b}else{if(this.eq(T))return e.eq(x)||e.eq(C)?T:e.eq(T)?x:(t=this.shr(1).div(e).shl(1),t.eq(y)?e.isNegative()?x:C:(r=this.sub(e.mul(t)),a=t.add(r.div(e)),a));if(e.eq(T))return this.unsigned?b:y;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();a=y}for(r=this;r.gte(e);){t=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var o=Math.ceil(Math.log(t)/Math.LN2),s=o<=48?1:u(2,o-48),d=c(t),p=d.mul(e);p.isNegative()||p.gt(r);)t-=s,d=c(t,this.unsigned),p=d.mul(e);d.isZero()&&(d=x),a=a.add(d),r=r.sub(p)}return a},E.div=E.divide,E.modulo=function(e){return i(e)||(e=f(e)),n?l((this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},E.mod=E.modulo,E.rem=E.modulo,E.not=function(){return l(~this.low,~this.high,this.unsigned)},E.and=function(e){return i(e)||(e=f(e)),l(this.low&e.low,this.high&e.high,this.unsigned)},E.or=function(e){return i(e)||(e=f(e)),l(this.low|e.low,this.high|e.high,this.unsigned)},E.xor=function(e){return i(e)||(e=f(e)),l(this.low^e.low,this.high^e.high,this.unsigned)},E.shiftLeft=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):l(0,this.low<<e-32,this.unsigned)},E.shl=E.shiftLeft,E.shiftRight=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):l(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},E.shr=E.shiftRight,E.shiftRightUnsigned=function(e){if(i(e)&&(e=e.toInt()),e&=63,e===0)return this;var t=this.high;if(e<32){var n=this.low;return l(n>>>e|t<<32-e,t>>>e,this.unsigned)}return l(e===32?t:t>>>e-32,0,this.unsigned)},E.shru=E.shiftRightUnsigned,E.shr_u=E.shiftRightUnsigned,E.toSigned=function(){return this.unsigned?l(this.low,this.high,!1):this},E.toUnsigned=function(){return this.unsigned?this:l(this.low,this.high,!0)},E.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},E.toBytesLE=function(){var e=this.high,t=this.low;return[t&255,t>>>8&255,t>>>16&255,t>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]},E.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,t>>>24,t>>>16&255,t>>>8&255,t&255]},r.fromBytes=function(e,t,n){return n?r.fromBytesLE(e,t):r.fromBytesBE(e,t)},r.fromBytesLE=function(e,t){return new r(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},r.fromBytesBE=function(e,t){return new r(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)}})))());
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
const vi=_i.default||_i;function yi(e){return vi.fromString(e,!0,16)}const bi=yi(`c3a5c85c97cb3127`),xi=yi(`b492b66fbe98f273`),O=yi(`9ae16a3b2f90404f`);function Si(e){return e.xor(e.shru(47))}function Ci(e,t,n){let r=e.slice(t,t+n);return vi.fromBytes(Array.from(r),!0,!0)}function k(e,t){return Ci(e,t,8)}function wi(e,t){return Ci(e,t,4)}function A(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function Ti(e,t,n=yi(`9ddfea08eb382d69`)){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let i=t.xor(r).mul(n);return i=i.xor(i.shru(47)),i=i.mul(n),i}function Ei(e,t,n,r,i,a){i=i.add(e),a=A(a.add(i).add(r),21);let o=i;return i=i.add(t),i=i.add(n),a=a.add(A(i,44)),[i.add(r),a.add(o)]}function Di(e,t,n,r){return Ei(k(e,t),k(e,t+8),k(e,t+16),k(e,t+24),n,r)}function Oi(e,t=e.length){if(t>=8){let n=O.add(t*2),r=k(e,0).add(O),i=k(e,t-8);return Ti(A(i,37).mul(n).add(r),A(r,25).add(i).mul(n),n)}if(t>=4){let n=O.add(t*2);return Ti(wi(e,0).shl(3).add(t),wi(e,t-4),n)}if(t>0){let n=e[0],r=e[t>>1],i=e[t-1],a=n+(r<<8),o=t+(i<<2);return Si(O.mul(a).xor(bi.mul(o))).mul(O)}return O}function ki(e,t=e.length){let n=O.add(t*2),r=k(e,0).mul(xi),i=k(e,8),a=k(e,t-8).mul(n),o=k(e,t-16).mul(O);return Ti(A(r.add(i),43).add(A(a,30)).add(o),r.add(A(i.add(O),18)).add(a),n)}function Ai(e,t=e.length){let n=O.add(t*2),r=k(e,0).mul(O),i=k(e,8),a=k(e,t-8).mul(n),o=k(e,t-16).mul(O),s=A(r.add(i),43).add(A(a,30)).add(o),c=Ti(s,r.add(A(i.add(O),18)).add(a),n),l=k(e,16).mul(n),u=k(e,24),d=s.add(k(e,t-32)).mul(n),f=c.add(k(e,t-24)).mul(n);return Ti(A(l.add(u),43).add(A(d,30)).add(f),l.add(A(u.add(r),18)).add(d),n)}function ji(e,t=e.length){let n=vi.fromNumber(81,!0);if(t<=32)return t<=16?Oi(e,t):ki(e,t);if(t<=64)return Ai(e,t);let r=n,i=n.mul(xi).add(113),a=Si(i.mul(O).add(113)).mul(O),o=[vi.UZERO,vi.UZERO],s=[vi.UZERO,vi.UZERO];r=r.mul(O).add(k(e,0));let c=0,l=(t-1>>6)*64,u=l+(t-1&63)-63;do r=A(r.add(i).add(o[0]).add(k(e,c+8)),37).mul(xi),i=A(i.add(o[1]).add(k(e,c+48)),42).mul(xi),r=r.xor(s[1]),i=i.add(o[0]).add(k(e,c+40)),a=A(a.add(s[0]),33).mul(xi),o=Di(e,c,o[1].mul(xi),r.add(s[0])),s=Di(e,c+32,a.add(s[1]),i.add(k(e,c+16))),[a,r]=[r,a],c+=64;while(c!==l);let d=xi.add(a.and(255).shl(1));return c=u,s[0]=s[0].add(t-1&63),o[0]=o[0].add(s[0]),s[0]=s[0].add(o[0]),r=A(r.add(i).add(o[0]).add(k(e,c+8)),37).mul(d),i=A(i.add(o[1]).add(k(e,c+48)),42).mul(d),r=r.xor(s[1].mul(9)),i=i.add(o[0].mul(9).add(k(e,c+40))),a=A(a.add(s[0]),33).mul(d),o=Di(e,c,o[1].mul(d),r.add(s[0])),s=Di(e,c+32,a.add(s[1]),i.add(k(e,c+16))),[a,r]=[r,a],Ti(Ti(o[0],s[0],d).add(Si(i).mul(bi)).add(a),Ti(o[1],s[1],d).add(r),d)}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
var Mi=/* @__PURE__ */ s({arraysEqual:()=>E,arraysEqualWithNull:()=>T,assert:()=>x,assertNonNegativeIntegerDimensions:()=>Ae,assertNonNull:()=>C,assertShapesMatch:()=>S,bytesFromStringArray:()=>ge,bytesPerElement:()=>he,checkConversionForErrors:()=>fe,clamp:()=>h,computeStrides:()=>Ce,convertBackendValuesAndArrayBuffer:()=>Ee,createScalarValue:()=>Ni,createShuffledIndices:()=>ie,decodeString:()=>zi,distSquared:()=>b,encodeString:()=>Ri,fetch:()=>Li,fingerPrint64:()=>ji,flatten:()=>Bi,getArrayFromDType:()=>de,getTypedArrayFromDType:()=>ue,hasEncodingLoss:()=>me,hexToLong:()=>yi,indexToLoc:()=>Me,inferDtype:()=>be,inferFromImplicitShape:()=>se,isBoolean:()=>ve,isFunction:()=>xe,isInt:()=>te,isNumber:()=>ye,isPromise:()=>Ne,isScalarShape:()=>ee,isString:()=>_e,isTypedArray:()=>j,isValidDtype:()=>pe,locToIndex:()=>je,makeOnesTypedArray:()=>De,makeZerosNestedTypedArray:()=>ke,makeZerosTypedArray:()=>Oe,nearestDivisor:()=>Se,nearestLargerEven:()=>g,now:()=>Ii,parseAxisParam:()=>ce,randUniform:()=>y,repeatedTry:()=>oe,rightPad:()=>ae,shuffle:()=>p,shuffleCombo:()=>m,sizeFromShape:()=>w,sizeToSquarishShape:()=>re,squeezeShape:()=>le,sum:()=>v,swap:()=>_,tanh:()=>ne,toNestedArray:()=>Te,toTypedArray:()=>Fi});function Ni(e,t){return t===`string`?Ri(e):Fi([e],t)}function Pi(e,t){return e instanceof Float32Array&&t===`float32`||e instanceof Int32Array&&t===`int32`||e instanceof Uint8Array&&t===`bool`}function Fi(e,t){if(t===`string`)throw Error(`Cannot convert a string[] to a TypedArray`);if(Array.isArray(e)&&(e=Bi(e)),D().getBool(`DEBUG`)&&fe(e,t),Pi(e,t))return e;if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`){let t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)Math.round(e[n])!==0&&(t[n]=1);return t}throw Error(`Unknown data type ${t}`)}function Ii(){return D().platform.now()}function Li(e,t){return D().platform.fetch(e,t)}function Ri(e,t=`utf-8`){return t||=`utf-8`,D().platform.encode(e,t)}function zi(e,t=`utf-8`){return t||=`utf-8`,D().platform.decode(e,t)}function j(e){return D().platform.isTypedArray==null?gi(e):D().platform.isTypedArray(e)}function Bi(e,t=[],n=!1){if(t??=[],typeof e==`boolean`||typeof e==`number`||typeof e==`string`||Ne(e)||e==null||j(e)&&n)t.push(e);else if(Array.isArray(e)||j(e))for(let r=0;r<e.length;++r)Bi(e[r],t,n);else{let r=-1;for(let t of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(t)&&(r=Math.max(r,Number(t)));for(let i=0;i<=r;i++)Bi(e[i],t,n)}return t}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Vi=class{constructor(e,t){this.backendTimer=e,this.logger=t,t??(this.logger=new Ui)}profileKernel(e,t,n){let r,i=()=>{r=n()},a,o=Ii();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(i);else{i();for(let e of r)e.dataSync();a=Promise.resolve({kernelMs:Ii()-o})}if(D().getBool(`CHECK_COMPUTATION_FOR_ERRORS`))for(let t=0;t<r.length;t++){let n=r[t];n.data().then(t=>{Hi(t,n.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:a.then(e=>e.kernelMs),extraInfo:a.then(e=>e.getExtraProfileInfo==null?``:e.getExtraProfileInfo())}}logKernelProfile(e){let{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:a}=e;n.forEach(e=>{Promise.all([e.data(),r,a]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],i,n[2])})})}};function Hi(e,t,n){if(t!==`float32`)return!1;for(let t=0;t<e.length;t++){let r=e[t];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}var Ui=class{logKernelProfile(e,t,n,r,i,a){let o=typeof r==`number`?ae(`${r}ms`,9):r.error,s=ae(e,25),c=t.rank,l=t.size,u=ae(t.shape.toString(),14),d=``;for(let e in i){let n=i[e];if(n!=null){let r=n.shape||t.shape,i=r.length;d+=`${e}: ${i}D ${i>0?r:``} `}}console.log(`%c${s}\t%c${o}\t%c${c}D ${u}\t%c${l}\t%c${d}\t%c${a}`,`font-weight:bold`,`color:red`,`color:blue`,`color: orange`,`color: green`,`color: steelblue`)}};
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
function Wi(e,t,n){let r={},i={};for(let e=0;e<t.length;e++)r[t[e].id]=!0;for(let n=0;n<e.length;n++){let a=e[n],o=a.inputs;for(let e in o){let n=o[e],s=!1;for(let e=0;e<t.length;e++)if(r[n.id]){a.outputs.forEach(e=>r[e.id]=!0),s=!0,i[a.id]=!0;break}if(s)break}}let a={};a[n.id]=!0;let o={};for(let t=e.length-1;t>=0;t--){let n=e[t],r=n.inputs;for(let e=0;e<n.outputs.length;e++)if(a[n.outputs[e].id]){for(let e in r)a[r[e].id]=!0,o[n.id]=!0;break}}let s=[];for(let t=0;t<e.length;t++){let n=e[t];if(i[n.id]&&o[n.id]){let e={};for(let t in n.inputs){let i=n.inputs[t];r[i.id]&&(e[t]=i)}let t=Object.assign({},n);t.inputs=e,t.outputs=n.outputs,s.push(t)}}return s}function Gi(e,t,n,r){for(let i=t.length-1;i>=0;i--){let a=t[i],o=[];if(a.outputs.forEach(t=>{let n=e[t.id];n==null?o.push(null):o.push(n)}),a.gradient==null)throw Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);let s=a.gradient(o);for(let t in a.inputs){if(!(t in s))throw Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(s)}.`);let i=n(()=>s[t]());if(i.dtype!==`float32`)throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${i.dtype}'`);let o=a.inputs[t];if(!E(i.shape,o.shape))throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${t}' has shape '${i.shape}', which does not match the shape of the input '${o.shape}'`);if(e[o.id]==null)e[o.id]=i;else{let t=e[o.id];e[o.id]=r(t,i),t.dispose()}}}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ki(e,t,n,r){let i=Ce(t),a=qi(e,t,n,i),o=t.length,s=Xi(e,t,n,i,a),c=[`Tensor`];return r&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${o}`),c.push(`  shape: [${t}]`),c.push(`  values:`)),c.push(s.map(e=>`    `+e).join(`
`)),c.join(`
`)}function qi(e,t,n,r){let i=w(t),a=r[r.length-1],o=Array(a).fill(0),s=t.length,c=n===`complex64`?Zi(e):e;if(s>1)for(let e=0;e<i/a;e++){let t=e*a;for(let e=0;e<a;e++)o[e]=Math.max(o[e],Ji(c[t+e],0,n).length)}return o}function Ji(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(7))} + ${parseFloat(e[1].toFixed(7))}j`:_e(e)?`'${e}'`:n===`bool`?Yi(e):parseFloat(e.toFixed(7)).toString(),ae(r,t)}function Yi(e){return e===0?`false`:`true`}function Xi(e,t,n,r,i,a=!0){let o=n===`complex64`?2:1,s=t[0],c=t.length;if(c===0)return n===`complex64`?[Ji(Zi(e)[0],0,n)]:n===`bool`?[Yi(e[0])]:[e[0].toString()];if(c===1){if(s>20){let t=3*o,r=Array.from(e.slice(0,t)),a=Array.from(e.slice((s-3)*o,s*o));return n===`complex64`&&(r=Zi(r),a=Zi(a)),[`[`+r.map((e,t)=>Ji(e,i[t],n)).join(`, `)+`, ..., `+a.map((e,t)=>Ji(e,i[s-3+t],n)).join(`, `)+`]`]}return[`[`+(n===`complex64`?Zi(e):Array.from(e)).map((e,t)=>Ji(e,i[t],n)).join(`, `)+`]`]}let l=t.slice(1),u=r.slice(1),d=r[0]*o,f=[];if(s>20){for(let t=0;t<3;t++){let r=t*d,a=r+d;f.push(...Xi(e.slice(r,a),l,n,u,i,!1))}f.push(`...`);for(let t=s-3;t<s;t++){let r=t*d,a=r+d;f.push(...Xi(e.slice(r,a),l,n,u,i,t===s-1))}}else for(let t=0;t<s;t++){let r=t*d,a=r+d;f.push(...Xi(e.slice(r,a),l,n,u,i,t===s-1))}let p=c===2?`,`:``;f[0]=`[`+(s>0?f[0]+p:``);for(let e=1;e<f.length-1;e++)f[e]=` `+f[e]+p;let m=`,
`;for(let e=2;e<c;e++)m+=`
`;return f[f.length-1]=` `+f[f.length-1]+`]`+(a?``:m),f}function Zi(e){let t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
var Qi=class{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=w(e),n!=null){let e=n.length;x(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if(t===`complex64`)throw Error(`complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).`);this.values=n||de(t,this.size),this.strides=Ce(e)}set(e,...t){t.length===0&&(t=[0]),x(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);let n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(let n of e){if(n<0||n>=this.shape[t]){let t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw Error(t)}t++}let n=e[e.length-1];for(let t=0;t<e.length-1;++t)n+=this.strides[t]*e[t];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];let t=Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return $i().makeTensor(this.values,this.shape,this.dtype)}};let $i=null,ea=null;function ta(e){$i=e}function na(e){ea=e}var M=class{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||`float32`,this.size=w(e),this.strides=Ce(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():`higher`}get rank(){return this.shape.length}async buffer(){let e=await this.data();return ea.buffer(this.shape,this.dtype,e)}bufferSync(){return ea.buffer(this.shape,this.dtype,this.dataSync())}async array(){let e=await this.data();return Te(this.shape,e,this.dtype===`complex64`)}arraySync(){return Te(this.shape,this.dataSync(),this.dtype===`complex64`)}async data(){this.throwIfDisposed();let e=$i().read(this.dataId);if(this.dtype===`string`){let t=await e;try{return t.map(e=>zi(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}}return e}dataToGPU(e){return this.throwIfDisposed(),$i().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();let e=$i().readSync(this.dataId);if(this.dtype===`string`)try{return e.map(e=>zi(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}return e}async bytes(){this.throwIfDisposed();let e=await $i().read(this.dataId);return this.dtype===`string`?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),$i().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw Error(`Tensor is disposed.`)}print(e=!1){return ea.print(this,e)}clone(){return this.throwIfDisposed(),ea.clone(this)}toString(e=!1){return Ki(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),ea.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),$i().makeVariable(this,e,t,n)}};Object.defineProperty(M,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function ra(){return We(`Tensor`,()=>M)}ra();var ia=class extends M{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!E(e.shape,this.shape))throw Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);$i().disposeTensor(this),this.dataId=e.dataId,$i().incRef(this,null)}dispose(){$i().disposeVariable(this),this.isDisposedInternal=!0}};Object.defineProperty(ia,Symbol.hasInstance,{value:e=>e instanceof M&&e.assign!=null&&e.assign instanceof Function});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
var aa;(function(e){e.R0=`R0`,e.R1=`R1`,e.R2=`R2`,e.R3=`R3`,e.R4=`R4`,e.R5=`R5`,e.R6=`R6`})(aa||={});var oa;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`int32`,e.complex64=`complex64`})(oa||={});var sa;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`bool`,e.complex64=`complex64`})(sa||={});var ca;(function(e){e.float32=`float32`,e.int32=`float32`,e.bool=`float32`,e.complex64=`complex64`})(ca||={});var la;(function(e){e.float32=`complex64`,e.int32=`complex64`,e.bool=`complex64`,e.complex64=`complex64`})(la||={});const ua={float32:ca,int32:oa,bool:sa,complex64:la};function da(e,t){if(e===`string`||t===`string`){if(e===`string`&&t===`string`)return`string`;throw Error(`Can not upcast ${e} with ${t}`)}return ua[e][t]}function fa(e){return da(e,`int32`)}function pa(e){return typeof e==`object`&&!!e&&`texture`in e&&e.texture instanceof WebGLTexture}function ma(e){return typeof GPUBuffer<`u`&&typeof e==`object`&&!!e&&`buffer`in e&&e.buffer instanceof GPUBuffer}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var ha=/* @__PURE__ */ s({assertTypesMatch:()=>ga,getTensorsInContainer:()=>va,isTensorInList:()=>_a,makeTypesMatch:()=>N});function N(e,t){if(e.dtype===t.dtype)return[e,t];let n=da(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function ga(e,t){x(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function _a(e,t){return t.some(t=>t.id===e.id)}function va(e){let t=[];return ya(e,t,/* @__PURE__ */ new Set),t}function ya(e,t,n){if(e==null)return;if(e instanceof M){t.push(e);return}if(!ba(e))return;let r=e;for(let e in r){let i=r[e];n.has(i)||(n.add(i),ya(i,t,n))}}function ba(e){return Array.isArray(e)||typeof e==`object`}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function xa(e){return e.kernelName!=null}var Sa=class{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=/* @__PURE__ */ new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(let e in this.registeredVariables)this.registeredVariables[e].dispose()}},Ca=class e{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Sa}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}get backend(){if(this.pendingBackendInit!=null)throw Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){let{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(e in this.registryFactory){let{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(ri(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;let{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new Vi(this.backendInstance),!0}setupRegisteredKernels(){li(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){li(e).forEach(t=>{t.disposeFunc!=null&&t.disposeFunc(this.registry[e])})}initializeBackend(e){let t=this.registryFactory[e];if(t==null)throw Error(`Cannot initialize backend ${e}, no registration found.`);try{let n=t.factory();if(n&&!(n instanceof d)&&typeof n.then==`function`){let t=++this.pendingBackendInitId,r=n.then(n=>t<this.pendingBackendInitId?!1:(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(n=>t<this.pendingBackendInitId?!1:(this.pendingBackendInit=null,ri(`Initialization of backend ${e} failed`),ri(n.stack||n.message),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(t){return ri(`Initialization of backend ${e} failed`),ri(t.stack||t.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw Error(`No backend found in registry.`);return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}moveData(e,t){let n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),a=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!=`function`)throw Error(`Please provide a function to tidy()`);t=e}else{if(typeof e!=`string`&&!(e instanceof String))throw Error(`When calling with two arguments, the first argument to tidy() must be a string`);if(typeof t!=`function`)throw Error(`When calling with two arguments, the 2nd argument to tidy() must be a function`);n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error(`Cannot return a Promise inside of tidy.`),r))}scopedRun(e,t,n){e();try{let e=n();return t(),e}catch(e){throw t(),e}}nextTensorId(){return e.nextTensorId++}nextVariableId(){return e.nextVariableId++}clone(e){let t=P.runKernel(on,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{let t={x:e};return P.runKernel(mt,t,{dtype:`float32`})}}),[],{}),t}runKernel(e,t,n){if(this.backendName??this.backend,si(e,this.backendName)==null)throw Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool(`IS_TEST`)}checkKernelForMemLeak(e,t,n){let r=this.backend.numDataIds(),i=0;n.forEach(e=>{i+=e.dtype===`complex64`?3:1});let a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-i-a;if(o>0)throw Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[],r=this.isTapeOn(),i=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let o;this.backendName??this.backend;let s,c=xa(e)?e.kernelName:this.state.activeScope==null?``:this.state.activeScope.name;if(xa(e)){let{kernelName:t,inputs:i,attrs:a}=e;this.backendName??this.backend;let c=si(t,this.backendName);x(c!=null,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{let e=this.backend.numDataIds();s=c.kernelFunc({inputs:i,attrs:a,backend:this.backend});let o=Array.isArray(s)?s:[s];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);let l=o.map(e=>e.rank==null?this.makeTensorFromTensorInfo(e):e);if(r){let e=this.getTensorsForGradient(t,i,l);n=this.saveTensorsForBackwardMode(e)}return l}}else{let{forwardFunc:t}=e,i=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{let e=this.backend.numDataIds();s=this.tidy(()=>t(this.backend,i));let n=Array.isArray(s)?s:[s];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,e,n),n}}let{inputs:l,attrs:u}=e,d=xa(e)?null:e.backwardsFunc,f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool(`DEBUG`)&&!this.state.profiling?t=o():(f=this.profiler.profileKernel(c,l,()=>o()),this.ENV.getBool(`DEBUG`)&&this.profiler.logKernelProfile(f),t=f.outputs)}),r&&this.addTapeNode(c,l,t,d,n,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(l).map(e=>l[e]==null?null:l[e].shape),outputShapes:t.map(e=>e.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(s)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){let r=ci(e);if(r!=null){let e=r.inputsToSave||[],i=r.outputsToSave||[],a;r.saveAllInputs?(x(Array.isArray(t),()=>`saveAllInputs is true, expected inputs to be an array.`),a=Object.keys(t).map(e=>t[e])):a=e.map(e=>t[e]);let o=n.filter((e,t)=>i[t]);return a.concat(o)}return[]}makeTensor(e,t,n,r){if(e==null)throw Error(`Values passed to engine.makeTensor() are null`);n||=`float32`,r||=this.backend;let i=e;n===`string`&&_e(e[0])&&(i=e.map(e=>Ri(e)));let a=r.write(i,t,n),o=new M(t,n,a,this.nextTensorId());if(this.trackTensor(o,r),n===`string`){let e=this.state.tensorInfo.get(a),t=ge(i);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){n||=`float32`;let i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){let{dataId:n,shape:r,dtype:i}=e,a=new M(r,i,n,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,n,r){n||=this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));let i=new ia(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype===`string`&&this.state.numStringTensors++;let n=0;e.dtype!==`complex64`&&e.dtype!==`string`&&(n=e.size*he(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof ia||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;let t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype===`string`&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!==`complex64`&&e.dtype!==`string`){let t=e.size*he(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(let e in this.state.registeredVariables){let t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){let e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons??=[],e.reasons.push(`Memory usage by string tensors is approximate (2 bytes per character)`)),e}async profile(e){this.state.profiling=!0;let t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(let e of this.state.activeProfile.kernels)e.kernelTimeMs=await e.kernelTimeMs,e.extraInfo=await e.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,a){let o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},s=ci(e);s!=null&&(r=s.gradFunc),r!=null&&(o.gradient=e=>(e=e.map((e,t)=>{if(e==null){let e=n[t],r=Oe(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],i,a))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){let t={track:[],name:`unnamed scope`,id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){let t=va(e),n=new Set(t.map(e=>e.id));for(let e=0;e<this.state.activeScope.track.length;e++){let t=this.state.activeScope.track[e];!t.kept&&!n.has(t.id)&&t.dispose()}let r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{!e.kept&&e.scopeId===r.id&&this.track(e)})}gradients(e,t,n,r=!1){if(x(t.length>0,()=>`gradients() received an empty list of xs.`),n!=null&&n.dtype!==`float32`)throw Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);let i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy(`forward`,e));x(i instanceof M,()=>`The result y returned by f() must be a tensor.`);let a=Wi(this.state.activeTape,t,i);if(!r&&a.length===0&&t.length>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.`);return this.tidy(`backward`,()=>{let e={};e[i.id]=n??wa(i.shape),Gi(e,a,e=>this.tidy(e),Ea);let r=t.map(t=>e[t.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(e=>{for(let t of e.saved)t.dispose()}),this.state.activeTape=null),{value:i,grads:r}})}customGrad(e){return x(xe(e),()=>`The f passed in customGrad(f) must be a function.`),(...t)=>{x(t.every(e=>e instanceof M),()=>`The args passed in customGrad(f)(x1, x2,...) must all be tensors`);let n,r={};return t.forEach((e,t)=>{r[t]=e}),this.runKernelFunc({forwardFunc:(r,i)=>(n=e(...t,i),x(n.value instanceof M,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),x(xe(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{let i=n.gradFunc(e,r),a=Array.isArray(i)?i:[i];x(a.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),x(a.every(e=>e instanceof M),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");let o={};return a.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){let t=Ii(),n=await this.backend.time(e);return n.wallMs=Ii()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Sa;for(let e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}};Ca.nextTensorId=0,Ca.nextVariableId=0;function wa(e){let t=De(w(e),`float32`);return P.makeTensor(t,e,`float32`)}function Ta(){let e=He();return e._tfengine??=new Ca(new Fe(e)),Be(e._tfengine.ENV),ta(()=>e._tfengine),e._tfengine}const P=Ta();function Ea(e,t){let n={a:e,b:t};return P.runKernel(`Add`,n)}var Da=/* @__PURE__ */ s({isBrowser:()=>Ma,isMobile:()=>ja,mockIsMobile:()=>Aa});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
function Oa(){return typeof navigator<`u`&&navigator!=null}let ka;function Aa(e){ka=e}function ja(e){if(ka!==void 0)return ka;if(e||Oa()){if(e||=navigator,e.product===`ReactNative`)return!0;let t=e.userAgent||e.vendor||(typeof window<`u`?window.opera:``);if(!t){let t=e;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Ma(){return typeof window<`u`&&window.document!=null||typeof WorkerGlobalScope<`u`}
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
const F=D();F.registerFlag(`DEBUG`,()=>!1,e=>{e&&console.warn(`Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.`)}),F.registerFlag(`IS_BROWSER`,()=>Ma()),F.registerFlag(`IS_NODE`,()=>typeof process<`u`&&process.versions!==void 0&&process.versions.node!==void 0),F.registerFlag(`IS_CHROME`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),F.registerFlag(`IS_SAFARI`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),F.registerFlag(`PROD`,()=>!1),F.registerFlag(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`,()=>F.getBool(`DEBUG`)),F.registerFlag(`DEPRECATION_WARNINGS_ENABLED`,()=>!0),F.registerFlag(`IS_TEST`,()=>!1),F.registerFlag(`CHECK_COMPUTATION_FOR_ERRORS`,()=>F.getBool(`DEBUG`)),F.registerFlag(`WRAP_TO_IMAGEBITMAP`,()=>!1),F.registerFlag(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`,()=>!1),F.registerFlag(`USE_SETTIMEOUTCUSTOM`,()=>!1);
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Na(e,t){let n=e;if(j(e))return t===`string`?[]:[e.length];if(pa(e)){let t=e.channels||`RGBA`;return[e.height,e.width*t.length]}if(ma(e))return[e.buffer.size/(t==null?4:he(t))];if(!Array.isArray(e))return[];let r=[];for(;Array.isArray(n)||j(n)&&t!==`string`;)r.push(n.length),n=n[0];return Array.isArray(e)&&D().getBool(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`)&&Pa(e,r,[]),r}function Pa(e,t,n){if(n||=[],!Array.isArray(e)&&!j(e)){x(t.length===0,()=>`Element arr[${n.join(`][`)}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}x(t.length>0,()=>`Element arr[${n.join(`][`)}] should be a primitive, but is an array of ${e.length} elements`),x(e.length===t[0],()=>`Element arr[${n.join(`][`)}] should have ${t[0]} elements, but has ${e.length} elements`);let r=t.slice(1);for(let t=0;t<e.length;++t)Pa(e[t],r,n.concat(t))}function Fa(e,t,n,r){if(e!==`string_or_numeric`){if(e==null)throw Error(`Expected dtype cannot be null.`);if(e!==`numeric`&&e!==t||e===`numeric`&&t===`string`)throw Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function I(e,t,n,r=`numeric`){if(e instanceof ra())return Fa(r,e.dtype,t,n),e;let i=be(e);if(i!==`string`&&[`bool`,`int32`,`float32`].indexOf(r)>=0&&(i=r),Fa(r,i,t,n),e==null||!j(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`){let r=e==null?`null`:e.constructor.name;throw Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}let a=Na(e,i);!j(e)&&!Array.isArray(e)&&(e=[e]);let o=i===`string`?Bi(e,[],!0):Fi(e,i);return P.makeTensor(o,a,i)}function Ia(e,t,n,r=`numeric`){if(!Array.isArray(e))throw Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,i)=>I(e,`${t}[${i}]`,n,r))}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
const La=`__op`;function L(e){let t=Object.keys(e);if(t.length!==1)throw Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0],r=e[n];n.endsWith(`_`)&&(n=n.substring(0,n.length-1)),n+=La;let i=(...e)=>{P.startScope(n);try{let t=r(...e);return Ne(t)&&console.error(`Cannot return a Promise inside of tidy.`),P.endScope(t),t}catch(e){throw P.endScope(null),e}};return Object.defineProperty(i,"name",{value:n,configurable:!0}),i}
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
function Ra(e,t){let n=I(e,`real`,`complex`),r=I(t,`imag`,`complex`);S(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);let i={real:n,imag:r};return P.runKernel(_t,i)}const za=/* @__PURE__ */ L({complex_:Ra});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ba(e,t,n,r){if(r==null)r=be(e);else if(r===`complex64`)throw Error(`Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).`);if(ma(e)||pa(e)){if(r!==`float32`&&r!==`int32`)throw Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return P.backend.createTensorFromGPUData(e,t||n,r)}if(!j(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`)throw Error(`values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray`);if(t!=null){Ae(t);let e=w(t),r=w(n);x(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let e=0;e<n.length;++e){let r=n[e],i=e!==n.length-1||r!==w(t.slice(e));x(n[e]===t[e]||!i,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!j(e)&&!Array.isArray(e)&&(e=[e]),t||=n,e=r===`string`?Bi(e,[],!0):Fi(e,r),P.makeTensor(e,t,r)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Va(e,t,n){return Ba(e,t,Na(e,n),n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
const Ha={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};var Ua=class e{static join(t){return new e(t).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(e=>j(e)?e.buffer:e),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){let r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);let i=t+r.byteLength;this.shards.push({buffer:r,start:t,end:i}),t=i}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0||(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e))return/* @__PURE__ */ new ArrayBuffer(0);let n=this.findShardForByte(e);if(n===-1)throw Error(`Could not find start shard for byte ${e}`);let r=t-e,i=new ArrayBuffer(r),a=new Uint8Array(i),o=0;for(let r=n;r<this.shards.length;r++){let n=this.shards[r],i=e+o-n.start,s=o,c=Math.min(t,n.end)-n.start,l=new Uint8Array(n.buffer,i,c-i);if(a.set(l,s),o+=l.length,t<n.end)break}return i}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(t){return e<t.start?-1:+(e>=t.end)}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;let n=Wa(this.shards,t);return n===-1?-1:(this.previousShardIndex=n,this.previousShardIndex)}};function Wa(e,t){let n=0,r=e.length;for(;n<=r;){let i=Math.floor((r-n)/2)+n,a=t(e[i]);if(a===0)return i;a<0?r=i:n=i+1}return-1}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ga(){D().set(`PROD`,!0)}function Ka(){D().set(`DEBUG`,!0)}function qa(){D().set(`DEPRECATION_WARNINGS_ENABLED`,!1),console.warn(`TensorFlow.js deprecation warnings have been disabled.`)}function Ja(e){D().getBool(`DEPRECATION_WARNINGS_ENABLED`)&&console.warn(e+` You can disable deprecation warnings with tf.disableDeprecationWarnings().`)}function Ya(){P.disposeVariables()}function Xa(){return P}function Za(){return P.memory()}function Qa(e){return P.profile(e)}function R(e,t){return P.tidy(e,t)}function z(e){va(e).forEach(e=>e.dispose())}function $a(e){return P.keep(e)}function eo(e){return P.time(e)}function to(e){return P.setBackend(e)}function no(){return P.ready()}function ro(){return P.backendName}function io(e){P.removeBackend(e)}function ao(e){return P.findBackend(e)}function oo(e){return P.findBackendFactory(e)}function so(e,t,n=1){return P.registerBackend(e,t,n)}function co(){return P.backend}function lo(e,t){D().setPlatform(e,t)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
async function uo(e,t){let n=[],r=[],i=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let a=0;a<i.length;++a){let o=i[a],s=Array.isArray(e)?e[a].tensor:e[o];if(s.dtype!==`float32`&&s.dtype!==`int32`&&s.dtype!==`bool`&&s.dtype!==`string`&&s.dtype!==`complex64`)throw Error(`Unsupported dtype in weight '${o}': ${s.dtype}`);let c={name:o,shape:s.shape,dtype:s.dtype};if(s.dtype===`string`){let e=new Promise(async e=>{let t=await s.bytes(),n=t.reduce((e,t)=>e+t.length,0)+4*t.length,r=new Uint8Array(n),i=0;for(let e=0;e<t.length;e++){let n=t[e],a=new Uint8Array(new Uint32Array([n.length]).buffer);r.set(a,i),i+=4,r.set(n,i),i+=n.length}e(r)});r.push(e)}else r.push(s.data());t!=null&&(c.group=t),n.push(c)}return{data:vo(await Promise.all(r)),specs:n}}function fo(e,t){let n=new Ua(e),r={},i=0;for(let e of t){let t=po(e,(e,t)=>n.slice(i+e,i+t));r[e.name]=ho(e,n.slice(i,i+t)),i+=t}return r}function po(e,t){let n=w(e.shape),r;if(`quantization`in e){let t=e.quantization;r=Ha[t.dtype]}else if(e.dtype===`string`){let e=0;for(let r=0;r<n;r++)e+=4+new Uint32Array(t(e,e+4))[0];return e}else r=Ha[e.dtype];return n*r}async function mo(e,t){let n=w(e.shape),r;if(`quantization`in e){let t=e.quantization;r=Ha[t.dtype]}else if(e.dtype===`string`){let e=0;for(let r=0;r<n;r++)e+=4+new Uint32Array(await t(e,e+4))[0];return e}else r=Ha[e.dtype];return n*r}function ho(e,t){let n=e.name,r=e.dtype,i=e.shape,a=w(i),o,s=0;if(`quantization`in e){let i=e.quantization;if(i.dtype===`uint8`||i.dtype===`uint16`){if(!(`min`in i&&`scale`in i))throw Error(`Weight ${e.name} with quantization ${i.dtype} doesn't have corresponding metadata min and scale.`)}else if(i.dtype===`float16`){if(r!==`float32`)throw Error(`Weight ${e.name} is quantized with ${i.dtype} which only supports weights of type float32 not ${r}.`)}else throw Error(`Weight ${e.name} has unknown quantization dtype ${i.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);let c=Ha[i.dtype],l=i.dtype===`uint8`?new Uint8Array(t):new Uint16Array(t);if(r===`float32`){if(i.dtype===`uint8`||i.dtype===`uint16`){o=new Float32Array(l.length);for(let e=0;e<l.length;e++){let t=l[e];o[e]=t*i.scale+i.min}}else if(i.dtype===`float16`)o=No()(l);else throw Error(`Unsupported quantization type ${i.dtype} for weight type float32.`)}else if(r===`int32`){if(i.dtype!==`uint8`&&i.dtype!==`uint16`)throw Error(`Unsupported quantization type ${i.dtype} for weight type int32.`);o=new Int32Array(l.length);for(let e=0;e<l.length;e++){let t=l[e];o[e]=Math.round(t*i.scale+i.min)}}else throw Error(`Unsupported dtype in weight '${n}': ${r}`);s+=a*c}else if(r===`string`){let n=w(e.shape);o=[];for(let e=0;e<n;e++){let e=new Uint32Array(t.slice(s,s+4))[0];s+=4;let n=new Uint8Array(t.slice(s,s+e));o.push(n),s+=e}}else{let e=Ha[r];if(r===`float32`)o=new Float32Array(t);else if(r===`int32`)o=new Int32Array(t);else if(r===`bool`)o=new Uint8Array(t);else if(r===`complex64`){o=new Float32Array(t);let e=new Float32Array(o.length/2),n=new Float32Array(o.length/2);for(let t=0;t<e.length;t++)e[t]=o[t*2],n[t]=o[t*2+1];let r=Va(e,i,`float32`),a=Va(n,i,`float32`),s=za(r,a);return r.dispose(),a.dispose(),s}else throw Error(`Unsupported dtype in weight '${n}': ${r}`);s+=a*e}return Va(o,i,r)}async function go(e,t,n){let r=new Uint8Array(t);for(;r.byteLength<n;){let{done:t,value:i}=await e.read();if(t&&i==null){let e=n-r.byteLength;throw Error(`Reader is done but ${e} bytes are still expected`)}let a=new Uint8Array(r.length+i.byteLength);a.set(r,0),a.set(new Uint8Array(i),r.length),r=a}return r.buffer}async function _o(e,t){let n={},r=e.getReader(),i=/* @__PURE__ */ new ArrayBuffer(0);for(let e of t){let t=await mo(e,async(e,t)=>(i=await go(r,i,t),i.slice(e,t)));i=await go(r,i,t);let a=i.slice(0,t);i=i.slice(t);let o=ho(e,a);if(n[e.name]=o,ro()===`webgpu`){let e=co();`uploadToGPU`in e&&w(o.shape)>=D().get(`WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD`)&&e.uploadToGPU(o.dataId)}}return n}function vo(e){if(e===null)throw Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0,n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});let r=new Uint8Array(t),i=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),i),i+=e.byteLength}),r.buffer}const yo=typeof Buffer<`u`&&(typeof Blob>`u`||typeof atob>`u`||typeof btoa>`u`);function bo(e){return yo?Buffer.byteLength(e,`utf8`):new Blob([e]).size}function xo(e){if(yo)return Buffer.from(e).toString(`base64`);let t=new Uint8Array(e),n=``;for(let e=0,r=t.length;e<r;e++)n+=String.fromCharCode(t[e]);return btoa(n)}function So(e){if(yo){let t=Buffer.from(e,`base64`);return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}let t=atob(e),n=new Uint8Array(t.length);for(let e=0;e<t.length;++e)n.set([t.charCodeAt(e)],e);return n.buffer}function Co(e){return Ua.join(e)}function wo(e){for(e=e.trim();e.endsWith(`/`);)e=e.slice(0,e.length-1);let t=e.split(`/`);return t[t.length-1]}function To(e,t){let n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function Eo(e,t,n){let r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(r.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw Error(`modelJSON has weightsManifest but weightSpecs is null`);if(!n)throw Error(`modelJSON has weightsManifest but weightData is null`);r.weightSpecs=t,r.weightData=n}return e.signature!=null&&(r.signature=e.signature),e.userDefinedMetadata!=null&&(r.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(r.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(r.initializerSignature=e.initializerSignature),r}async function Do(e,t){let n,r;return e.weightsManifest!=null&&([n,r]=await t(e.weightsManifest)),Eo(e,n,r)}function Oo(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`Expected JSON model topology, received ArrayBuffer.`);return{dateSaved:/* @__PURE__ */ new Date,modelTopologyType:`JSON`,modelTopologyBytes:e.modelTopology==null?0:bo(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:bo(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new Ua(e.weightData).byteLength}}function ko(e){let t=[];for(let n of e)t.push(...n.weights);return t}function Ao(){let e=e=>{let t=e<<13,n=0;for(;!(t&8388608);)n-=8388608,t<<=1;return t&=-8388609,n+=947912704,t|n},t=/* @__PURE__ */ new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let e=1024;e<2048;e++)t[e]=939524096+(e-1024<<13);return t}function jo(){let e=/* @__PURE__ */ new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Mo(){let e=/* @__PURE__ */ new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function No(){let e=Ao(),t=jo(),n=Mo();return r=>{let i=/* @__PURE__ */ new ArrayBuffer(4*r.length),a=new Uint32Array(i);for(let i=0;i<r.length;i++){let o=r[i],s=e[n[o>>10]+(o&1023)]+t[o>>10];a[i]=s}return new Float32Array(i)}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Po=class e{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return e.instance??=new e,e.instance}static registerSaveRouter(t){e.getInstance().saveRouters.push(t)}static registerLoadRouter(t){e.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return e.getHandlers(t,`save`)}static getLoadHandlers(t,n){return e.getHandlers(t,`load`,n)}static getHandlers(t,n,r){let i=[];return(n===`load`?e.getInstance().loadRouters:e.getInstance().saveRouters).forEach(e=>{let n=e(t,r);n!==null&&i.push(n)}),i}};const Fo=e=>Po.registerSaveRouter(e),Io=e=>Po.registerLoadRouter(e),Lo=e=>Po.getSaveHandlers(e),Ro=(e,t)=>Po.getLoadHandlers(e,t),zo=`tensorflowjs`,Bo=`models_store`,Vo=`model_info_store`
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
;function Ho(){if(!D().getBool(`IS_BROWSER`))throw Error(`Failed to obtain IndexedDB factory because the current environmentis not a web browser.`);let e=typeof window>`u`?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw Error(`The current browser does not appear to support IndexedDB.`);return t}function Uo(e){let t=e.result;t.createObjectStore(Bo,{keyPath:`modelPath`}),t.createObjectStore(Vo,{keyPath:`modelPath`})}var Wo=class{constructor(e){if(this.indexedDB=Ho(),e==null||!e)throw Error(`For IndexedDB, modelPath must not be null, undefined or empty.`);this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{let r=this.indexedDB.open(zo,1);r.onupgradeneeded=()=>Uo(r),r.onsuccess=()=>{let i=r.result;if(t==null){let t=i.transaction(Bo,`readonly`),r=t.objectStore(Bo).get(this.modelPath);r.onsuccess=()=>{if(r.result==null)return i.close(),n(/* @__PURE__ */ Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(i.close(),n(r.error)),t.oncomplete=()=>i.close()}else{t.weightData=Ua.join(t.weightData);let r=Oo(t),a=i.transaction(Vo,`readwrite`),o=a.objectStore(Vo),s;try{s=o.put({modelPath:this.modelPath,modelArtifactsInfo:r})}catch(e){return n(e)}let c;s.onsuccess=()=>{c=i.transaction(Bo,`readwrite`);let s=c.objectStore(Bo),l;try{l=s.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r})}catch(e){return n(e)}l.onsuccess=()=>e({modelArtifactsInfo:r}),l.onerror=e=>{o=a.objectStore(Vo);let t=o.delete(this.modelPath);t.onsuccess=()=>(i.close(),n(l.error)),t.onerror=e=>(i.close(),n(l.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}}},r.onerror=e=>n(r.error)})}};Wo.URL_SCHEME=`indexeddb://`;const Go=e=>D().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(Wo.URL_SCHEME)?Ko(e.slice(Wo.URL_SCHEME.length)):null;Po.registerSaveRouter(Go),Po.registerLoadRouter(Go);function Ko(e){return new Wo(e)}function qo(e){return e.startsWith(Wo.URL_SCHEME)?e.slice(Wo.URL_SCHEME.length):e}var Jo=class{constructor(){this.indexedDB=Ho()}async listModels(){return new Promise((e,t)=>{let n=this.indexedDB.open(zo,1);n.onupgradeneeded=()=>Uo(n),n.onsuccess=()=>{let r=n.result,i=r.transaction(Vo,`readonly`),a=i.objectStore(Vo).getAll();a.onsuccess=()=>{let t={};for(let e of a.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},a.onerror=e=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){return e=qo(e),new Promise((t,n)=>{let r=this.indexedDB.open(zo,1);r.onupgradeneeded=()=>Uo(r),r.onsuccess=()=>{let i=r.result,a=i.transaction(Vo,`readwrite`),o=a.objectStore(Vo),s=o.get(e),c;s.onsuccess=()=>{if(s.result==null)return i.close(),n(/* @__PURE__ */ Error(`Cannot find model with path '${e}' in IndexedDB.`));{let r=o.delete(e),a=()=>{c=i.transaction(Bo,`readwrite`);let r=c.objectStore(Bo).delete(e);r.onsuccess=()=>t(s.result.modelArtifactsInfo),r.onerror=e=>n(s.error)};r.onsuccess=a,r.onerror=e=>(a(),i.close(),n(s.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}},r.onerror=e=>n(r.error)})}};
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
const Yo=`tensorflowjs_models`;function Xo(e){return{info:[Yo,e,`info`].join(`/`),topology:[Yo,e,`model_topology`].join(`/`),weightSpecs:[Yo,e,`weight_specs`].join(`/`),weightData:[Yo,e,`weight_data`].join(`/`),modelMetadata:[Yo,e,`model_metadata`].join(`/`)}}function Zo(e){for(let t of Object.values(e))window.localStorage.removeItem(t)}function Qo(e){let t=e.split(`/`);if(t.length<3)throw Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(`/`)}function $o(e){return e.startsWith(es.URL_SCHEME)?e.slice(es.URL_SCHEME.length):e}var es=class{constructor(e){if(!D().getBool(`IS_BROWSER`)||typeof window>`u`||window.localStorage===void 0)throw Error(`The current environment does not support local storage.`);if(this.LS=window.localStorage,e==null||!e)throw Error(`For local storage, modelPath must not be null, undefined or empty.`);this.modelPath=e,this.keys=Xo(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);{let t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=Oo(e),i=Ua.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,xo(i));let a={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature==null?void 0:e.signature,userDefinedMetadata:e.userDefinedMetadata==null?void 0:e.userDefinedMetadata,modelInitializer:e.modelInitializer==null?void 0:e.modelInitializer,initializerSignature:e.initializerSignature==null?void 0:e.initializerSignature,trainingConfig:e.trainingConfig==null?void 0:e.trainingConfig};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw Zo(this.keys),Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){let e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!==`JSON`)throw Error(`BrowserLocalStorage does not support loading non-JSON model topology yet.`);let t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;let r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;let i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){let e=JSON.parse(i);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,e.signature!=null&&(t.signature=e.signature),e.userDefinedMetadata!=null&&(t.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(t.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(t.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(t.trainingConfig=e.trainingConfig)}let a=this.LS.getItem(this.keys.weightData);if(a==null)throw Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=So(a),t}};es.URL_SCHEME=`localstorage://`;const ts=e=>D().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(es.URL_SCHEME)?ns(e.slice(es.URL_SCHEME.length)):null;Po.registerSaveRouter(ts),Po.registerLoadRouter(ts);function ns(e){return new es(e)}var rs=class{constructor(){x(D().getBool(`IS_BROWSER`),()=>`Current environment is not a web browser`),x(typeof window>`u`||window.localStorage!==void 0,()=>`Current browser does not appear to support localStorage`),this.LS=window.localStorage}async listModels(){let e={};for(let t=0;t<this.LS.length;++t){let n=this.LS.key(t);if(n.startsWith(`tensorflowjs_models/`)&&n.endsWith(`/info`)){let t=Qo(n);e[t]=JSON.parse(this.LS.getItem(n))}}return e}async removeModel(e){e=$o(e);let t=Xo(e);if(this.LS.getItem(t.info)==null)throw Error(`Cannot find model at path '${e}'`);let n=JSON.parse(this.LS.getItem(t.info));return Zo(t),n}},is=class e{constructor(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
this.managers={}}static getInstance(){return e.instance??=new e,e.instance}static registerManager(t,n){x(t!=null,()=>`scheme must not be undefined or null.`),t.endsWith(`://`)&&(t=t.slice(0,t.indexOf(`://`))),x(t.length>0,()=>`scheme must not be an empty string.`);let r=e.getInstance();x(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){let n=e.getInstance().managers[t];if(n==null)throw Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(e.getInstance().managers)}};function as(e){if(e.indexOf(`://`)===-1)throw Error(`The url string provided does not contain a scheme. Supported schemes are: ${is.getSchemes().join(`,`)}`);return{scheme:e.split(`://`)[0],path:e.split(`://`)[1]}}async function os(e,t,n=!1){x(e!==t,()=>`Old path and new path are the same: '${e}'`);let r=Po.getLoadHandlers(e);x(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),x(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);let i=r[0],a=Po.getSaveHandlers(t);x(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),x(a.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);let o=a[0],s=as(e).scheme,c=as(e).path,l=s===as(e).scheme,u=await i.load();n&&l&&await is.getManager(s).removeModel(c);let d=await o.save(u);return n&&!l&&await is.getManager(s).removeModel(c),d.modelArtifactsInfo}async function ss(){let e=is.getSchemes(),t={};for(let n of e){let e=await is.getManager(n).listModels();for(let r in e){let i=n+`://`+r;t[i]=e[r]}}return t}async function cs(e){let t=as(e);return is.getManager(t.scheme).removeModel(t.path)}async function ls(e,t){return os(e,t,!1)}async function us(e,t){return os(e,t,!0)}
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
var ds=class{constructor(){this.messageName=`setTimeoutCustom`,this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder??=new TextEncoder,this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>`u`||!D().getBool(`USE_SETTIMEOUTCUSTOM`)){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},`*`)},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener(`message`,e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();let t=this.functionRefs[e.data.index];t(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return gi(e)}};if(D().get(`IS_BROWSER`)){D().setPlatform(`browser`,new ds);try{is.registerManager(es.URL_SCHEME,new rs)}catch{}try{is.registerManager(Wo.URL_SCHEME,new Jo)}catch{}}var fs=/* @__PURE__ */ o(((e,t)=>{t.exports={}}));
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
const ps={importFetch:()=>fs()};let ms;var hs=class{constructor(){this.util=fs(),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return D().global.fetch==null?(ms??=ps.importFetch(),ms(e,t)):D().global.fetch(e,t)}now(){let e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?``:new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}};D().get(`IS_NODE`)&&!D().get(`IS_BROWSER`)&&D().setPlatform(`node`,new hs);
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
function gs(e,t=`float32`,n){return t||=`float32`,Ae(e),new Qi(e,t,n)}
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
function _s(e,t){let n=I(e,`x`,`cast`);if(!pe(t))throw Error(`Failed to cast to unknown dtype ${t}`);if(t===`string`&&n.dtype!==`string`||t!==`string`&&n.dtype===`string`)throw Error(`Only strings can be casted to strings`);let r={x:n},i={dtype:t};return P.runKernel(mt,r,i)}const B=/* @__PURE__ */ L({cast_:_s});
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
function vs(e){let t={x:I(e,`x`,`clone`,`string_or_numeric`)};return P.runKernel(on,t)}const ys=/* @__PURE__ */ L({clone_:vs});
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
function bs(e,t=!1){console.log(e.toString(t))}
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
Ta(),na({buffer:gs,cast:B,clone:ys,print:bs});
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
function xs(e,t){let n=I(e,`a`,`add`),r=I(t,`b`,`add`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel(`Add`,i)}const V=/* @__PURE__ */ L({add_:xs});
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
function Ss(e,t){let n=I(e,`a`,`floorDiv`),r=I(t,`b`,`floorDiv`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel($t,i)}const Cs=/* @__PURE__ */ L({floorDiv_:Ss});
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
function ws(e,t){let n=I(e,`a`,`div`),r=I(t,`b`,`div`);if([n,r]=N(n,r),n.dtype===`int32`&&r.dtype===`int32`)return Cs(n,r);let i={a:n,b:r};return P.runKernel(Vt,i,{})}const H=/* @__PURE__ */ L({div_:ws});
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
function Ts(e,t){let n=I(e,`a`,`mul`),r=I(t,`b`,`mul`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel(Ln,i)}const U=/* @__PURE__ */ L({mul_:Ts});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Es(e){let t=I(e,`x`,`abs`);if(t.dtype===`complex64`){let e={x:t};return P.runKernel(vt,e)}{let e={x:t};return P.runKernel(`Abs`,e)}}const W=/* @__PURE__ */ L({abs_:Es});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ds(e){let t={x:I(e,`x`,`acos`)};return P.runKernel(Ke,t)}const Os=/* @__PURE__ */ L({acos_:Ds});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function ks(e){let t={x:I(e,`x`,`acosh`)};return P.runKernel(qe,t)}const As=/* @__PURE__ */ L({acosh_:ks});
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
function js(e){x(Array.isArray(e),()=>`The argument passed to tf.addN() must be a list of tensors`),x(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);let t=e.map((e,t)=>I(e,`tensors${t}`,`addN`)),n=t[0];t.forEach(e=>{if(e.dtype!==n.dtype)throw Error(`All tensors passed to tf.addN() must have the same dtype`)}),t.forEach(e=>{if(!E(e.shape,n.shape))throw Error(`All tensors passed to tf.addN() must have the same shape`)});let r=t;return P.runKernel(Ye,r)}const Ms=/* @__PURE__ */ L({addN_:js});
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
function Ns(e,t=null,n=!1){let r={x:I(e,`x`,`all`,`bool`)},i={axis:t,keepDims:n};return P.runKernel(`All`,r,i)}const Ps=/* @__PURE__ */ L({all_:Ns});
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
function Fs(e,t=null,n=!1){let r={x:I(e,`x`,`any`,`bool`)},i={axis:t,keepDims:n};return P.runKernel(`Any`,r,i)}const Is=/* @__PURE__ */ L({any_:Fs});
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
function Ls(e,t=0){let n={x:I(e,`x`,`argMax`)},r={axis:t};return P.runKernel(Qe,n,r)}const Rs=/* @__PURE__ */ L({argMax_:Ls});
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
function zs(e,t=0){let n={x:I(e,`x`,`argMin`)},r={axis:t};return P.runKernel($e,n,r)}const Bs=/* @__PURE__ */ L({argMin_:zs});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Vs(e){let t={x:I(e,`x`,`asin`)};return P.runKernel(et,t)}const Hs=/* @__PURE__ */ L({asin_:Vs});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Us(e){let t={x:I(e,`x`,`asinh`)};return P.runKernel(tt,t)}const Ws=/* @__PURE__ */ L({asinh_:Us});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Gs(e){let t={x:I(e,`x`,`atan`)};return P.runKernel(nt,t)}const Ks=/* @__PURE__ */ L({atan_:Gs});
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
function qs(e,t){let n=I(e,`a`,`atan2`),r=I(t,`b`,`atan2`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel(it,i)}const Js=/* @__PURE__ */ L({atan2_:qs});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ys(e){let t={x:I(e,`x`,`atanh`)};return P.runKernel(rt,t)}const Xs=/* @__PURE__ */ L({atanh_:Ys});
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
function Zs(e,t,n,r,i=`NHWC`,a){let o=e[3];return ec(e,[...t,o],n,a,r,null,null,mc(i))}function Qs(e,t,n,r,i,a,o=`channelsLast`){let[s,c]=ac(t),l;if(o===`channelsLast`)l=[s,c,e[3],e[3]];else if(o===`channelsFirst`)l=[s,c,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return ec(e,l,n,r,i,a,!1,o)}function $s(e,t,n,r,i,a,o=`NDHWC`){let[s,c,l]=oc(t),u,d;if(o===`NDHWC`)d=`channelsLast`,u=[s,c,l,e[4],e[4]];else if(o===`NCDHW`)d=`channelsFirst`,u=[s,c,l,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return tc(e,u,n,r,i,!1,d,a)}function ec(e,t,n,r,i,a,o=!1,s=`channelsLast`){let[c,l,u,d]=[-1,-1,-1,-1];if(s===`channelsLast`)[c,l,u,d]=e;else if(s===`channelsFirst`)[c,d,l,u]=e;else throw Error(`Unknown dataFormat ${s}`);let[f,p,,m]=t,[h,g]=ac(n),[_,v]=ac(r),y=sc(f,_),b=sc(p,v),{padInfo:x,outHeight:S,outWidth:C}=cc(i,l,u,h,g,y,b,a,s),w=o?m*d:m,ee;return s===`channelsFirst`?ee=[c,w,S,C]:s===`channelsLast`&&(ee=[c,S,C,w]),{batchSize:c,dataFormat:s,inHeight:l,inWidth:u,inChannels:d,outHeight:S,outWidth:C,outChannels:w,padInfo:x,strideHeight:h,strideWidth:g,filterHeight:f,filterWidth:p,effectiveFilterHeight:y,effectiveFilterWidth:b,dilationHeight:_,dilationWidth:v,inShape:e,outShape:ee,filterShape:t}}function tc(e,t,n,r,i,a=!1,o=`channelsLast`,s){let[c,l,u,d,f]=[-1,-1,-1,-1,-1];if(o===`channelsLast`)[c,l,u,d,f]=e;else if(o===`channelsFirst`)[c,f,l,u,d]=e;else throw Error(`Unknown dataFormat ${o}`);let[p,m,h,,g]=t,[_,v,y]=oc(n),[b,x,S]=oc(r),C=sc(p,b),w=sc(m,x),ee=sc(h,S),{padInfo:T,outDepth:E,outHeight:te,outWidth:ne}=lc(i,l,u,d,_,v,y,C,w,ee,s),re=a?g*f:g,ie;return o===`channelsFirst`?ie=[c,re,E,te,ne]:o===`channelsLast`&&(ie=[c,E,te,ne,re]),{batchSize:c,dataFormat:o,inDepth:l,inHeight:u,inWidth:d,inChannels:f,outDepth:E,outHeight:te,outWidth:ne,outChannels:re,padInfo:T,strideDepth:_,strideHeight:v,strideWidth:y,filterDepth:p,filterHeight:m,filterWidth:h,effectiveFilterDepth:C,effectiveFilterHeight:w,effectiveFilterWidth:ee,dilationDepth:b,dilationHeight:x,dilationWidth:S,inShape:e,outShape:ie,filterShape:t}}function nc(e,t,n,r,i){r??=ic(e,t,n);let a=e[0],o=e[1];return[uc((a-t+2*r)/n+1,i),uc((o-t+2*r)/n+1,i)]}function rc(e,t,n,r,i,a){i??=ic(e,t[0],r[0]);let o=[0,0,0,n];for(let n=0;n<3;n++)e[n]+2*i>=t[n]&&(o[n]=uc((e[n]-t[n]+2*i)/r[n]+1,a));return o}function ic(e,t,n,r=1){let i=sc(t,r);return Math.floor((e[0]*(n-1)-n+i)/2)}function ac(e){return typeof e==`number`?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function oc(e){return typeof e==`number`?[e,e,e]:e}function sc(e,t){return t<=1?e:e+(e-1)*(t-1)}function cc(e,t,n,r,i,a,o,s,c){let l,u,d;if(typeof e==`number`){l={top:e,bottom:e,left:e,right:e,type:e===0?`VALID`:`NUMBER`};let i=nc([t,n],a,r,e,s);u=i[0],d=i[1]}else if(e===`same`){u=Math.ceil(t/r),d=Math.ceil(n/i);let e=Math.max(0,(u-1)*r+a-t),s=Math.max(0,(d-1)*i+o-n),c=Math.floor(e/2),f=e-c,p=Math.floor(s/2);l={top:c,bottom:f,left:p,right:s-p,type:`SAME`}}else if(e===`valid`)l={top:0,bottom:0,left:0,right:0,type:`VALID`},u=Math.ceil((t-a+1)/r),d=Math.ceil((n-o+1)/i);else if(typeof e==`object`){let f=c===`channelsLast`?e[1][0]:e[2][0],p=c===`channelsLast`?e[1][1]:e[2][1],m=c===`channelsLast`?e[2][0]:e[3][0],h=c===`channelsLast`?e[2][1]:e[3][1];l={top:f,bottom:p,left:m,right:h,type:f===0&&p===0&&m===0&&h===0?`VALID`:`EXPLICIT`},u=uc((t-a+f+p)/r+1,s),d=uc((n-o+m+h)/i+1,s)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:u,outWidth:d}}function lc(e,t,n,r,i,a,o,s,c,l,u){let d,f,p,m;if(e===`valid`&&(e=0),typeof e==`number`){d={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?`VALID`:`NUMBER`};let h=rc([t,n,r,1],[s,c,l],1,[i,a,o],e,u);f=h[0],p=h[1],m=h[2]}else if(e===`same`){f=Math.ceil(t/i),p=Math.ceil(n/a),m=Math.ceil(r/o);let e=(f-1)*i+s-t,u=(p-1)*a+c-n,h=(m-1)*o+l-r,g=Math.floor(e/2),_=e-g,v=Math.floor(u/2),y=u-v,b=Math.floor(h/2);d={top:v,bottom:y,left:b,right:h-b,front:g,back:_,type:`SAME`}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:d,outDepth:f,outHeight:p,outWidth:m}}function uc(e,t){if(!t)return Math.trunc(e);switch(t){case`round`:return Math.round(e);case`ceil`:return Math.ceil(e);case`floor`:return Math.floor(e);default:throw Error(`Unknown roundingMode ${t}`)}}function dc(e){let[t,n,r]=ac(e);return t===1&&n===1&&r===1}function fc(e,t){return dc(e)||dc(t)}function pc(e){return ac(e).every(e=>e>0)}function mc(e){if(e===`NHWC`)return`channelsLast`;if(e===`NCHW`)return`channelsFirst`;throw Error(`Unknown dataFormat ${e}`)}function hc(e,t,n){if(n!=null){if(typeof t==`string`)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t==`number`)x(te(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t==`object`)t.forEach(t=>{t.forEach(t=>{x(te(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}
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
function gc(e,t){let n={x:I(e,`x`,`reshape`,`string_or_numeric`)},r={shape:t};return P.runKernel(rr,n,r)}const G=/* @__PURE__ */ L({reshape_:gc});
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
function _c(e,t,n,r,i){let a=I(e,`x`,`avgPool`,`float32`);x(fc(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let o=a,s=!1;a.rank===3&&(s=!0,o=G(a,[1,a.shape[0],a.shape[1],a.shape[2]])),x(o.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),hc(`avgPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=P.runKernel(at,c,l);return u=B(u,a.dtype),s?G(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const vc=/* @__PURE__ */ L({avgPool_:_c});
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
function yc(e,t,n,r,i,a=`NDHWC`){let o=I(e,`x`,`avgPool3d`,`float32`),s=o,c=!1;o.rank===4&&(c=!0,s=G(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),x(s.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${s.rank}.`),x(a===`NDHWC`,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),x(typeof n==`number`&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),hc(`avgPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=P.runKernel(st,l,u);return d=B(d,s.dtype),c?G(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const bc=/* @__PURE__ */ L({avgPool3d_:yc});
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
function xc(e,t=0){x(e.length>=1,()=>`Pass at least one tensor to concat`);let n=Ia(e,`tensors`,`concat`,`string_or_numeric`);if(n[0].dtype===`complex64`&&n.forEach(e=>{if(e.dtype!==`complex64`)throw Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${e.dtype}. `)}),n.length===1)return ys(n[0]);let r=n,i={axis:t};return P.runKernel(yt,r,i)}const K=/* @__PURE__ */ L({concat_:xc});
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
function Sc(e,t,n=!1,r=!1){let i=I(e,`a`,`matMul`),a=I(t,`b`,`matMul`);[i,a]=N(i,a);let o={a:i,b:a},s={transposeA:n,transposeB:r};return P.runKernel(lt,o,s)}const q=/* @__PURE__ */ L({matMul_:Sc});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Cc(e){let t={x:I(e,`x`,`sigmoid`,`float32`)};return P.runKernel(xr,t)}const wc=/* @__PURE__ */ L({sigmoid_:Cc});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Tc(e,t,n){let r=I(e,`x`,`slice`,`string_or_numeric`);if(r.rank===0)throw Error(`Slicing scalar is not possible`);let i={x:r},a={begin:t,size:n};return P.runKernel(_r,i,a)}const J=/* @__PURE__ */ L({slice_:Tc});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ec(e){let t={x:I(e,`x`,`tanh`,`float32`)};return P.runKernel(Hr,t)}const Dc=/* @__PURE__ */ L({tanh_:Ec});
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
function Oc(e,t,n,r,i,a){let o=I(e,`forgetBias`,`basicLSTMCell`),s=I(t,`lstmKernel`,`basicLSTMCell`),c=I(n,`lstmBias`,`basicLSTMCell`),l=I(r,`data`,`basicLSTMCell`),u=I(i,`c`,`basicLSTMCell`),d=I(a,`h`,`basicLSTMCell`),f=K([l,d],1),p=q(f,s),m=V(p,c),h=m.shape[0],g=m.shape[1]/4,_=[h,g],v=J(m,[0,0],_),y=J(m,[0,g],_),b=J(m,[0,g*2],_),x=J(m,[0,g*3],_),S=V(U(wc(v),Dc(y)),U(u,wc(V(o,b))));return[S,U(Dc(S),wc(x))]}const kc=/* @__PURE__ */ L({basicLSTMCell_:Oc});
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
function Ac(e,t,n){let r=I(e,`x`,`batchToSpaceND`),i=t.reduce((e,t)=>e*t);x(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),x(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),x(r.shape[0]%i===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(` * `)} === ${i}`);let a={x:r},o={blockShape:t,crops:n};return P.runKernel(ut,a,o)}const jc=/* @__PURE__ */ L({batchToSpaceND_:Ac});function Mc(e){let t;return t=e.rank===0||e.rank===1?G(e,[1,1,1,e.size]):e.rank===2?G(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?G(e,[1,e.shape[0],e.shape[1],e.shape[2]]):e,t}
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
function Nc(e,t,n,r,i,a){a??=.001;let o=I(e,`x`,`batchNorm`),s=I(t,`mean`,`batchNorm`),c=I(n,`variance`,`batchNorm`),l;i!=null&&(l=I(i,`scale`,`batchNorm`));let u;r!=null&&(u=I(r,`offset`,`batchNorm`)),x(s.rank===c.rank,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),x(u==null||s.rank===u.rank,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),x(l==null||s.rank===l.rank,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let d={x:Mc(o),scale:l,offset:u,mean:s,variance:c},f={varianceEpsilon:a},p=P.runKernel(en,d,f);return G(p,o.shape)}const Pc=/* @__PURE__ */ L({batchNorm_:Nc});function Fc(e,t,n,r,i,a){let o=I(e,`x`,`batchNorm`),s=I(t,`mean`,`batchNorm`),c=I(n,`variance`,`batchNorm`),l;i!=null&&(l=I(i,`scale`,`batchNorm`));let u;return r!=null&&(u=I(r,`offset`,`batchNorm`)),x(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),x(s.rank===2||s.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${s.rank}.`),x(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),l!=null&&x(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),u!=null&&x(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${u.rank}.`),Pc(o,s,c,u,l,a)}const Ic=/* @__PURE__ */ L({batchNorm2d_:Fc});function Lc(e,t,n,r,i,a){let o=I(e,`x`,`batchNorm`),s=I(t,`mean`,`batchNorm`),c=I(n,`variance`,`batchNorm`),l;i!=null&&(l=I(i,`scale`,`batchNorm`));let u;return r!=null&&(u=I(r,`offset`,`batchNorm`)),x(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),x(s.rank===3||s.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${s.rank}.`),x(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),l!=null&&x(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),u!=null&&x(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${u.rank}.`),Pc(o,s,c,u,l,a)}const Rc=/* @__PURE__ */ L({batchNorm3d_:Lc});function zc(e,t,n,r,i,a){let o=I(e,`x`,`batchNorm`),s=I(t,`mean`,`batchNorm`),c=I(n,`variance`,`batchNorm`),l;i!=null&&(l=I(i,`scale`,`batchNorm`));let u;return r!=null&&(u=I(r,`offset`,`batchNorm`)),x(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),x(s.rank===4||s.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${s.rank}.`),x(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),l!=null&&x(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),u!=null&&x(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${u.rank}.`),Pc(o,s,c,u,l,a)}const Bc=/* @__PURE__ */ L({batchNorm4d_:zc});
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
function Vc(e,t,n){let r=I(e,`x`,`bincount`),i=I(t,`weights`,`bincount`);x(r.dtype===`int32`,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),x(n>=0,()=>`size must be non-negative, but got ${n}.`),x(i.size===r.size||i.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${i.shape}.`);let a={x:r,weights:i},o={size:n};return P.runKernel(dt,a,o)}const Hc=/* @__PURE__ */ L({bincount_:Vc});
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
function Uc(e,t){let n=I(e,`x`,`bitwiseAnd`),r=I(t,`y`,`bitwiseAnd`);if(!E(n.shape,r.shape))throw Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${r.shape}`);if(n.dtype!==`int32`||r.dtype!==`int32`)throw Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${r.dtype}`);let i={a:n,b:r};return P.runKernel(ft,i)}const Wc=/* @__PURE__ */ L({bitwiseAnd_:Uc});
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
function Gc(e,t){let n=I(e,`s0`,`broadcastArgs`,`int32`),r=I(t,`s1`,`broadcastArgs`,`int32`);if(n.rank!==1)throw Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(r.rank!==1)throw Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);let i={s0:n,s1:r};return P.runKernel(pt,i)}const Kc=/* @__PURE__ */ L({broadcastArgs_:Gc});
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
function qc(e,t){let n=I(e,`broadcastTo`,`x`),r=n.shape;if(Ae(t),t.length<n.rank)throw Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){let e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=G(n,e)}let i=n.shape,a=Array.from(t);for(let e=t.length-1;e>=0;e--)if(i[e]===t[e])a[e]=1;else if(n.shape[e]!==1)throw Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(a.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length===0)return ys(n);let o={x:n},s={reps:a};return P.runKernel(Ur,o,s)}const Jc=/* @__PURE__ */ L({broadcastTo_:qc});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Yc(e){let t={x:I(e,`x`,`ceil`,`float32`)};return P.runKernel(ht,t)}const Xc=/* @__PURE__ */ L({ceil_:Yc});
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
function Zc(e,t,n){Ae(e),n||=be(t);let r={shape:e,value:t,dtype:n};return P.runKernel(Xt,{},r)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Qc(e,t,n){let r=I(e,`x`,`clipByValue`);if(x(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Zc(r.shape,t,r.dtype);let i={x:r},a={clipValueMin:t,clipValueMax:n};return P.runKernel(gt,i,a)}const $c=/* @__PURE__ */ L({clipByValue_:Qc});function el(e){return K(e,0)}const tl=/* @__PURE__ */ L({concat1d_:el});function nl(e,t){return K(e,t)}const rl=/* @__PURE__ */ L({concat2d_:nl});function il(e,t){return K(e,t)}const al=/* @__PURE__ */ L({concat3d_:il});function ol(e,t){return K(e,t)}const sl=/* @__PURE__ */ L({concat4d_:ol});
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
function cl(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=I(e,`x`,`conv2d`,`float32`),c=I(t,`filter`,`conv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=G(s,[1,s.shape[0],s.shape[1],s.shape[2]])),x(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),x(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),hc(`conv2d`,r,o);let d=i===`NHWC`?l.shape[3]:l.shape[1];x(d===c.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${c.shape[2]}.`),x(fc(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),x(pc(a),()=>`Error in conv2D: Dilated rates should be larger than 0.`),x(pc(n),()=>`Error in conv2D: Strides should be larger than 0.`);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},m=P.runKernel(bt,f,p);return u?G(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const ll=/* @__PURE__ */ L({conv2d_:cl});function ul(e,t,n,r,i=`NWC`,a=1,o){let s=I(e,`x`,`conv1d`),c=I(t,`filter`,`conv1d`),l=s,u=!1;s.rank===2&&(u=!0,l=G(s,[1,s.shape[0],s.shape[1]])),x(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),x(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),hc(`conv1d`,r,o),x(l.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${c.shape[1]}.`),x(fc(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),x(pc(a),()=>`Error in conv1D: Dilated rates should be larger than 0.`),x(pc(n),()=>`Error in conv1D: Stride should be larger than 0.`),x(i===`NWC`,()=>`Error in conv1d: got dataFormat of ${i} but only NWC is currently supported.`);let d=G(c,[1,c.shape[0],c.shape[1],c.shape[2]]),f=G(l,[l.shape[0],1,l.shape[1],l.shape[2]]),p=ll(f,d,[1,n],r,`NHWC`,[1,a],o);return u?G(p,[p.shape[2],p.shape[3]]):G(p,[p.shape[0],p.shape[2],p.shape[3]])}const dl=/* @__PURE__ */ L({conv1d_:ul});
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
function fl(e,t,n,r,i,a=`NHWC`,o){x(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let s=e,c=t,l=!1;t.rank===3&&(l=!0,c=G(t,[1,t.shape[0],t.shape[1],t.shape[2]]),s=[1,e[0],e[1],e[2]]),x(s.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${s.length}.`),x(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),x(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);let u=a===`NHWC`?s[3]:s[1],d=a===`NHWC`?c.shape[3]:c.shape[1];x(u===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[2]}.`),x(d===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${d}) must match output depth for filter ${n.shape[3]}.`),hc(`conv2dDerInput`,i,o);let f={dy:c,filter:n},p={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,inputShape:s},m=P.runKernel(St,f,p);return l?G(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const pl=/* @__PURE__ */ L({conv2DBackpropInput_:fl});function ml(e,t,n,r,i,a){let o=I(e,`x`,`conv2dTranspose`),s=I(t,`filter`,`conv2dTranspose`);return pl(n,o,s,r,i,`NHWC`,a)}const hl=/* @__PURE__ */ L({conv2dTranspose_:ml});
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
function gl(e,t,n,r,i=`NDHWC`,a=[1,1,1]){let o=I(e,`x`,`conv3d`),s=I(t,`filter`,`conv3d`),c=o,l=!1;o.rank===4&&(l=!0,c=G(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),x(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),x(s.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${s.rank}.`),x(c.shape[4]===s.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${s.shape[3]}.`),x(fc(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),x(i===`NDHWC`,()=>`Error in conv3d: got dataFormat of ${i} but only NDHWC is currently supported.`),x(pc(a),()=>`Error in conv3D: Dilated rates should be larger than 0.`),x(pc(n),()=>`Error in conv3D: Strides should be larger than 0.`);let u={x:c,filter:s},d={strides:n,pad:r,dataFormat:i,dilations:a},f=P.runKernel(Ct,u,d);return l?G(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const _l=/* @__PURE__ */ L({conv3d_:gl});
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
function vl(e,t,n,r,i){x(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,s=!1;t.rank===4&&(s=!0,o=G(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);let c=a[4],l=o.shape[4];x(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),x(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),x(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),x(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),x(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);let u={dy:o,filter:n},d={pad:i,strides:r,inputShape:a},f=P.runKernel(Tt,u,d);return s?G(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const yl=/* @__PURE__ */ L({conv3DBackpropInput_:vl});function bl(e,t,n,r,i){let a=I(e,`x`,`conv3dTranspose`),o=I(t,`filter`,`conv3dTranspose`);return yl(n,a,o,r,i)}const xl=/* @__PURE__ */ L({conv3dTranspose_:bl});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Sl(e){let t={x:I(e,`x`,`cos`,`float32`)};return P.runKernel(`Cos`,t)}const Cl=/* @__PURE__ */ L({cos_:Sl});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function wl(e){let t={x:I(e,`x`,`cosh`,`float32`)};return P.runKernel(Dt,t)}const Tl=/* @__PURE__ */ L({cosh_:wl});
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
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
function El(e,t=0,n=!1,r=!1){let i={x:I(e,`x`,`cumprod`)},a={axis:t,exclusive:n,reverse:r};return P.runKernel(Ot,i,a)}const Dl=/* @__PURE__ */ L({cumprod_:El});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ol(e,t=0,n=!1,r=!1){let i={x:I(e,`x`,`cumsum`)},a={axis:t,exclusive:n,reverse:r};return P.runKernel(kt,i,a)}const kl=/* @__PURE__ */ L({cumsum_:Ol});
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
function Al(e,t,n,r=!1){let i=I(e,`x`,`denseBincount`),a=I(t,`weights`,`denseBincount`);x(i.dtype===`int32`,()=>`Error in denseBincount: input dtype must be int32, but got ${i.dtype}`),x(i.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${i.rank}.`),x(n>=0,()=>`size must be non-negative, but got ${n}.`),x(a.size===i.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${i.shape}, weights shape: ${a.shape}.`);let o={x:i,weights:a},s={size:n,binaryOutput:r};return P.runKernel(jt,o,s)}const jl=/* @__PURE__ */ L({denseBincount_:Al});
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
function Ml(e,t,n=`NHWC`){let r=I(e,`x`,`depthToSpace`,`float32`),i=n===`NHWC`?r.shape[1]:r.shape[2],a=n===`NHWC`?r.shape[2]:r.shape[3],o=n===`NHWC`?r.shape[3]:r.shape[1];x(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),x(i*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${i} and ${t}  for depthToSpace with input shape
    ${r.shape}`),x(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${r.shape}`),x(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);let s={x:r},c={blockSize:t,dataFormat:n};return P.runKernel(Mt,s,c)}const Nl=/* @__PURE__ */ L({depthToSpace_:Ml});
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
function Pl(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=I(e,`x`,`depthwiseConv2d`,`float32`),c=I(t,`filter`,`depthwiseConv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=G(s,[1,s.shape[0],s.shape[1],s.shape[2]])),x(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),x(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);let d=i===`NHWC`?l.shape[3]:l.shape[1];x(d===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${d}) must match the inChannels dimension in filter ${c.shape[2]}.`),hc(`depthwiseConv2d`,r,o);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},m=P.runKernel(Nt,f,p);return u?G(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const Fl=/* @__PURE__ */ L({depthwiseConv2d_:Pl});
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
function Il(e){let t={x:I(e,`x`,`diag`)};return P.runKernel(It,t)}const Ll=/* @__PURE__ */ L({diag_:Il});
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
function Rl(e,t,n,r,i=[1,1],a=`NHWC`){let o=I(e,`x`,`dilation2d`),s=I(t,`filter`,`dilation2d`);x(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),x(s.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${s.rank}.`),x(a===`NHWC`,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let c=o,l=!1;o.rank===3&&(c=G(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0),x(c.shape[3]===s.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${c.shape[3]} vs ${s.shape[2]}`);let u={x:c,filter:s},d={strides:n,pad:r,dilations:i},f=P.runKernel(Lt,u,d);return l?G(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const zl=/* @__PURE__ */ L({dilation2d_:Rl});var Bl=/* @__PURE__ */ s({assertAndGetBroadcastShape:()=>Y,getBroadcastDims:()=>Vl,getReductionAxes:()=>Hl});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
function Vl(e,t){let n=e.length,r=[];for(let i=0;i<n;i++){let a=n-1-i,o=e[a]||1;(t[t.length-1-i]||1)>1&&o===1&&r.unshift(a)}return r}function Hl(e,t){let n=[];for(let r=0;r<t.length;r++){let i=e[e.length-r-1],a=t.length-r-1,o=t[a];(i==null||i===1&&o>1)&&n.unshift(a)}return n}function Y(e,t){let n=Math.max(e.length,t.length),r=Array(n);for(let i=0;i<n;i++){let a=e[e.length-i-1];a??=1;let o=t[t.length-i-1];if(o??=1,a===1)r[n-i-1]=o;else if(o===1)r[n-i-1]=a;else if(a!==o){let n=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(n)}else r[n-i-1]=a}return r}
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
function Ul(e,t){let n=I(e,`a`,`equal`,`string_or_numeric`),r=I(t,`b`,`equal`,`string_or_numeric`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Kt,i)}const Wl=/* @__PURE__ */ L({equal_:Ul});
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
function Gl(e,t,n){let r=I(t,`a`,`where`),i=I(n,`b`,`where`),a=I(e,`condition`,`where`,`bool`),o=Y(Y(a.shape,r.shape),i.shape),s={condition:Jc(a,o),t:Jc(r,o),e:Jc(i,o)};return P.runKernel(hr,s)}const Kl=/* @__PURE__ */ L({where_:Gl});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function ql(e){let t={x:I(e,`x`,`zerosLike`)};return P.runKernel(Xr,t)}const Jl=/* @__PURE__ */ L({zerosLike_:ql});
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
function Yl(e,t){let n=I(e,`a`,`div`),r=I(t,`b`,`div`);[n,r]=N(n,r);let i=H(n,r),a=Jl(i),o=Wl(r,a);return Kl(o,a,i)}const Xl=/* @__PURE__ */ L({divNoNan_:Yl});
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
function Zl(e,t){let n=I(e,`t1`,`dot`),r=I(t,`t2`,`dot`);x((n.rank===1||n.rank===2)&&(r.rank===1||r.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);let i=n.rank===1?n.size:n.shape[1],a=r.rank===1?r.size:r.shape[0];if(x(i===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${i} and ${a}.`),n.rank===1&&r.rank===1){let e=G(n,[1,-1]),t=G(r,[-1,1]),i=q(e,t);return G(i,[])}if(n.rank===1&&r.rank===2){let e=G(n,[1,-1]),t=G(r,[r.shape[0],r.shape[1]]),i=q(e,t);return G(i,[i.size])}if(n.rank===2&&r.rank===1){let e=G(r,[-1,1]),t=q(n,e);return G(t,[t.size])}{let e=G(r,[r.shape[0],r.shape[1]]);return q(n,e)}}const Ql=/* @__PURE__ */ L({dot_:Zl});
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
function $l(e,...t){let n=t.map((e,t)=>I(e,`tensors${t}`,`einsum`)),r={equation:e};return P.runKernel(Ht,n,r)}const eu=/* @__PURE__ */ L({einsum_:$l});
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
function tu(e){let t={x:I(e,`x`,`elu`,`float32`)};return P.runKernel(`Elu`,t)}const nu=/* @__PURE__ */ L({elu_:tu});
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
function ru(e,t){let n=I(e,`x`,`ensureShape`,`string_or_numeric`);if(!T(n.shape,t))throw Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}const iu=/* @__PURE__ */ L({ensureShape_:ru});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function au(e){let t=I(e,`x`,`erf`);x(t.dtype===`int32`||t.dtype===`float32`,()=>"Input dtype must be `int32` or `float32`."),t.dtype===`int32`&&(t=B(t,`float32`));let n={x:t};return P.runKernel(`Erf`,n)}const ou=/* @__PURE__ */ L({erf_:au});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
function su(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function cu(e,t,n){let r=e.length+t.length,i=[],a=0,o=0;for(let s=0;s<r;s++)n.indexOf(s)===-1?i.push(e[a++]):i.push(t[o++]);return i}function lu(e,t){let n=[],r=e.length;for(let i=0;i<r;i++)t.indexOf(i)===-1&&n.push(e[i]);return[n,t.map(t=>e[t])]}function uu(e,t){return cu(e,t.map(e=>1),t)}function du(e,t,n){x(su(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function fu(e,t){if(su(e,t))return null;let n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(e=>n.push(e)),n}function pu(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function mu(e,t){let n=[];for(let r=t-e;r<t;++r)n.push(r);return n}
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
function hu(e,t=null,n=!1){let r={x:I(e,`x`,`max`)},i={reductionIndices:t,keepDims:n};return P.runKernel(`Max`,r,i)}const gu=/* @__PURE__ */ L({max_:hu});
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
function _u(e,t=null,n=!1){let r={x:I(e,`x`,`min`)},i={axis:t,keepDims:n};return P.runKernel(`Min`,r,i)}const vu=/* @__PURE__ */ L({min_:_u});
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
function yu(e,t){let n=I(e,`base`,`pow`),r=I(t,`exp`,`pow`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel(`Pow`,i)}const bu=/* @__PURE__ */ L({pow_:yu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function X(e,t){if((j(e)&&t!==`string`||Array.isArray(e))&&t!==`complex64`)throw Error(`Error creating a new Scalar: value must be a primitive (number|boolean|string)`);if(t===`string`&&j(e)&&!(e instanceof Uint8Array))throw Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Ba(e,[],[],t)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function xu(e){let t={x:I(e,`x`,`sqrt`,`float32`)};return P.runKernel(Cr,t)}const Su=/* @__PURE__ */ L({sqrt_:xu});
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
function Cu(e){let t=I(e,`x`,`square`);return P.runKernel(`Square`,{x:t},{})}const wu=/* @__PURE__ */ L({square_:Cu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Tu(e,t=null,n=!1){let r=I(e,`x`,`sum`);r.dtype===`bool`&&(r=B(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return P.runKernel(`Sum`,i,a)}const Z=/* @__PURE__ */ L({sum_:Tu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Eu(e,t=`euclidean`,n=null,r=!1){e=I(e,`x`,`norm`);let i=Du(e,t,n),a=i.shape;if(r){let t=ce(n,e.shape);a=uu(i.shape,t)}return G(i,a)}function Du(e,t,n=null){if(e.rank===0)return W(e);if(e.rank!==1&&n===null)return Du(G(e,[-1]),t,n);if(e.rank===1||typeof n==`number`||Array.isArray(n)&&n.length===1){if(t===1)return Z(W(e),n);if(t===1/0)return gu(W(e),n);if(t===-1/0)return vu(W(e),n);if(t===`euclidean`||t===2)return Su(Z(bu(W(e),X(2,`int32`)),n));throw Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return gu(Z(W(e),n[0]),n[1]-1);if(t===1/0)return gu(Z(W(e),n[1]),n[0]);if(t===-1/0)return vu(Z(W(e),n[1]),n[0]);if(t===`fro`||t===`euclidean`)return Su(Z(wu(e),n));throw Error(`Error in norm: invalid ord value: ${t}`)}throw Error(`Error in norm: invalid axis: ${n}`)}const Ou=/* @__PURE__ */ L({norm_:Eu});
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
function ku(e,t=null,n=!1){return Ou(e,`euclidean`,t,n)}const Au=/* @__PURE__ */ L({euclideanNorm_:ku});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function ju(e){let t={x:I(e,`x`,`exp`)};return P.runKernel(`Exp`,t)}const Mu=/* @__PURE__ */ L({exp_:ju});
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
function Nu(e,t=0){let n=I(e,`x`,`expandDims`,`string_or_numeric`);x(t<=n.rank,()=>`Axis must be <= rank of the tensor`);let r={input:n},i={dim:t};return P.runKernel(Jt,r,i)}const Pu=/* @__PURE__ */ L({expandDims_:Nu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Fu(e){let t={x:I(e,`x`,`expm1`)};return P.runKernel(Yt,t)}const Iu=/* @__PURE__ */ L({expm1_:Fu});
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
function Lu(e,t){let n=I(e,`x`,`tile`,`string_or_numeric`);x(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);let r={x:n},i={reps:t};return P.runKernel(Ur,r,i)}const Ru=/* @__PURE__ */ L({tile_:Lu});
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
function zu(e,t,n,r=`float32`){t??=e;let i=gs([e,t],r),a=e<=t?e:t;for(let e=0;e<a;++e)i.set(1,e,e);let o=G(i.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return Ru(Pu(o,0),[n[0],1,1]);if(n.length===2)return Ru(Pu(Pu(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return Ru(Pu(Pu(Pu(o,0),0),0),[n[0],n[1],n[2],1,1]);throw Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const Bu=/* @__PURE__ */ L({eye_:zu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Vu(e){let t={x:I(e,`x`,`floor`,`float32`)};return P.runKernel(Qt,t)}const Hu=/* @__PURE__ */ L({floor_:Vu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Uu(e,t,n=0,r=0){let i={x:I(e,`x`,`gather`),indices:I(t,`indices`,`gather`,`int32`)},a={axis:n,batchDims:r};return P.runKernel(tn,i,a)}const Wu=/* @__PURE__ */ L({gather_:Uu});
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
function Gu(e,t){let n=I(e,`a`,`greater`,`string_or_numeric`),r=I(t,`b`,`greater`,`string_or_numeric`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(rn,i)}const Ku=/* @__PURE__ */ L({greater_:Gu});
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
function qu(e,t){let n=I(e,`a`,`greaterEqual`,`string_or_numeric`),r=I(t,`b`,`greaterEqual`,`string_or_numeric`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(an,i)}const Ju=/* @__PURE__ */ L({greaterEqual_:qu});
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
function Yu(e){let t={input:I(e,`input`,`imag`)};return P.runKernel(cn,t)}const Xu=/* @__PURE__ */ L({imag_:Yu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Zu(e){let t={x:I(e,`x`,`isFinite`)};return P.runKernel(ln,t)}const Qu=/* @__PURE__ */ L({isFinite_:Zu});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function $u(e){let t={x:I(e,`x`,`isInf`)};return P.runKernel(un,t)}const ed=/* @__PURE__ */ L({isInf_:$u});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function td(e){let t={x:I(e,`x`,`isNaN`)};return P.runKernel(dn,t)}const nd=/* @__PURE__ */ L({isNaN_:td});
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
function rd(e,t=.2){let n={x:I(e,`x`,`leakyRelu`)},r={alpha:t};return P.runKernel(fn,n,r)}const id=/* @__PURE__ */ L({leakyRelu_:rd});
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
function ad(e,t){let n=I(e,`a`,`less`,`string_or_numeric`),r=I(t,`b`,`less`,`string_or_numeric`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(pn,i)}const od=/* @__PURE__ */ L({less_:ad});
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
function sd(e,t){let n=I(e,`a`,`lessEqual`,`string_or_numeric`),r=I(t,`b`,`lessEqual`,`string_or_numeric`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(mn,i)}const cd=/* @__PURE__ */ L({lessEqual_:sd});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function ld(e,t,n){if(n<=0)throw Error(`The number of values should be positive.`);let r={start:e,stop:t,num:n};return P.runKernel(hn,{},r)}
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
function ud(e,t=5,n=1,r=1,i=.5){let a=I(e,`x`,`localResponseNormalization`);x(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),x(te(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,s=!1;a.rank===3&&(s=!0,o=G(a,[1,a.shape[0],a.shape[1],a.shape[2]]));let c={x:o},l={depthRadius:t,bias:n,alpha:r,beta:i},u=P.runKernel(`LRN`,c,l);return s?G(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const dd=/* @__PURE__ */ L({localResponseNormalization_:ud});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function fd(e){let t={x:I(e,`x`,`log`,`float32`)};return P.runKernel(`Log`,t)}const pd=/* @__PURE__ */ L({log_:fd});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function md(e){let t={x:I(e,`x`,`log1p`)};return P.runKernel(_n,t)}const hd=/* @__PURE__ */ L({log1p_:md});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function gd(e){return x(xe(e),()=>`The f passed in grad(f) must be a function`),(t,n)=>{let r=I(t,`x`,`tf.grad`,`string_or_numeric`),i=n==null?null:I(n,`dy`,`tf.grad`);return P.tidy(()=>{let{value:t,grads:n}=P.gradients(()=>e(r),[r],i);return i!=null&&S(t.shape,i.shape,`The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)`),Sd(n),n[0]})}}function _d(e){return x(xe(e),()=>`The f passed in grads(f) must be a function`),(t,n)=>{x(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");let r=Ia(t,`args`,`tf.grads`,`string_or_numeric`),i=n==null?null:I(n,`dy`,`tf.grads`);return P.tidy(()=>{let{value:t,grads:n}=P.gradients(()=>e(...r),r,i);return i!=null&&S(t.shape,i.shape,`The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])`),Sd(n),n})}}function vd(e){return x(xe(e),()=>`The f passed in valueAndGrad(f) must be a function`),(t,n)=>{x(t instanceof M,()=>`The x passed in valueAndGrad(f)(x) must be a tensor`),x(n==null||n instanceof M,()=>`The dy passed in valueAndGrad(f)(x, dy) must be a tensor`);let{grads:r,value:i}=P.gradients(()=>e(t),[t],n);return Sd(r),{grad:r[0],value:i}}}function yd(e){return x(xe(e),()=>`The f passed in valueAndGrads(f) must be a function`),(t,n)=>{x(Array.isArray(t)&&t.every(e=>e instanceof M),()=>`The args passed in valueAndGrads(f)(args) must be array of tensors`),x(n==null||n instanceof M,()=>`The dy passed in valueAndGrads(f)(args, dy) must be a tensor`);let r=P.gradients(()=>e(...t),t,n);return n!=null&&S(r.value.shape,n.shape,`The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])`),Sd(r.grads),r}}function bd(e,t){x(xe(e),()=>`The f passed in variableGrads(f) must be a function`),x(t==null||Array.isArray(t)&&t.every(e=>e instanceof ia),()=>`The varList passed in variableGrads(f, varList) must be an array of variables`);let n=t!=null;if(!n){t=[];for(let e in P.registeredVariables)t.push(P.registeredVariables[e])}let r=n?t.filter(e=>!e.trainable):null,i=t.length;t=t.filter(e=>e.trainable),x(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${i} variables is trainable.`);let{value:a,grads:o}=P.gradients(e,t,null,!0);x(o.some(e=>e!=null),()=>`Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize().`),x(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);let s={};return t.forEach((e,t)=>{o[t]!=null&&(s[e.name]=o[t])}),r?.forEach(e=>s[e.name]=null),{value:a,grads:s}}function xd(e){return P.customGrad(e)}function Sd(e){if(e.filter(e=>e==null).length>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Cd(e){let t={x:I(e,`x`,`neg`)};return P.runKernel(`Neg`,t)}const wd=/* @__PURE__ */ L({neg_:Cd});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Td(e){let t={x:I(e,`x`,`softplus`)};return P.runKernel(Sr,t)}const Ed=/* @__PURE__ */ L({softplus_:Td});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Dd(e){let t=I(e,`x`,`logSigmoid`);return xd(e=>({value:wd(Ed(wd(e))),gradFunc:t=>U(t,wc(wd(e)))}))(t)}const Od=/* @__PURE__ */ L({logSigmoid_:Dd});
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
function kd(e,t){let n=I(e,`a`,`sub`),r=I(t,`b`,`sub`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel(`Sub`,i)}const Q=/* @__PURE__ */ L({sub_:kd});
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
function Ad(e,t=-1){let n=I(e,`logits`,`logSoftmax`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return xd((e,n)=>{let r=gu(e,t,!0),i=Q(e,r),a=Q(B(i,`float32`),pd(Z(Mu(i),t,!0)));return n([a]),{value:a,gradFunc:(e,n)=>{let[r]=n,i=Mu(r);return Q(e,U(Z(e,t,!0),i))}}})(n)}const jd=/* @__PURE__ */ L({logSoftmax_:Ad});
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
function Md(e,t=null,n=!1){let r=I(e,`x`,`logSumExp`),i=ce(t,r.shape),a=gu(r,i,!0),o=Q(r,a),s=Mu(o),c=Z(s,i),l=pd(c),u=V(G(a,l.shape),l);if(n){let e=uu(u.shape,i);return G(u,e)}return u}const Nd=/* @__PURE__ */ L({logSumExp_:Md});
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
function Pd(e,t){let n=I(e,`a`,`logicalAnd`,`bool`),r=I(t,`b`,`logicalAnd`,`bool`);Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(vn,i)}const Fd=/* @__PURE__ */ L({logicalAnd_:Pd});
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
function Id(e){let t={x:I(e,`x`,`logicalNot`,`bool`)};return P.runKernel(yn,t)}const Ld=/* @__PURE__ */ L({logicalNot_:Id});
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
function Rd(e,t){let n=I(e,`a`,`logicalOr`,`bool`),r=I(t,`b`,`logicalOr`,`bool`);Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(bn,i)}const zd=/* @__PURE__ */ L({logicalOr_:Rd});
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
function Bd(e,t){let n=I(e,`a`,`logicalXor`,`bool`),r=I(t,`b`,`logicalXor`,`bool`);return Y(n.shape,r.shape),Fd(zd(e,t),Ld(Fd(e,t)))}const Vd=/* @__PURE__ */ L({logicalXor_:Bd}),Hd=2147483648
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
;function Ud(e,t,n=`left`){let r=I(e,`sortedSequence`,`searchSorted`),i=I(t,`values`,`searchSorted`),a=r.shape[r.shape.length-1],o=i.shape[i.shape.length-1],s=G(r,[-1,a]),c=G(i,[-1,o]);if(s.rank<2)throw Error(`Sorted input argument must be at least 2-dimensional`);if(s.shape[0]!==c.shape[0])throw Error(`Leading dimension of 'sortedSequence' and 'values' must match.`);if(w(c.shape)>=Hd)throw Error(`values tensor size must less than ${Hd}`);if(s.shape[1]>=Hd)throw Error(`trailing dim_size must less than ${Hd} for int32 output type, was ${s.shape[1]}`);let l={sortedSequence:s,values:c},u={side:n};return P.runKernel(mr,l,u)}const Wd=/* @__PURE__ */ L({searchSorted_:Ud});
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
function Gd(e,t){return Wd(e,t,`left`)}
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
function Kd(e,t,n,r,i){let a=I(e,`x`,`maxPool`),o=a,s=!1;a.rank===3&&(s=!0,o=G(a,[1,a.shape[0],a.shape[1],a.shape[2]])),x(o.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),x(fc(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),hc(`maxPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=P.runKernel(En,c,l);return s?G(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const qd=/* @__PURE__ */ L({maxPool_:Kd});
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
function Jd(e,t=[1,1,1],n,r,i,a=`NDHWC`){let o=I(e,`x`,`maxPool3d`),s=o,c=!1;o.rank===4&&(c=!0,s=G(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),x(s.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${s.rank}.`),x(a===`NDHWC`,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),hc(`maxPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=P.runKernel(On,l,u);return c?G(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const Yd=/* @__PURE__ */ L({maxPool3d_:Jd});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Xd(e,t,n,r,i=!1){let a={x:I(e,`x`,`maxPoolWithArgmax`)},o={filterSize:t,strides:n,pad:r,includeBatchInIndex:i},s=P.runKernel(An,a,o);return{result:s[0],indexes:s[1]}}const Zd=/* @__PURE__ */ L({maxPoolWithArgmax_:Xd});
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
function Qd(e,t){let n=I(e,`a`,`maximum`),r=I(t,`b`,`maximum`);[n,r]=N(n,r),n.dtype===`bool`&&(n=B(n,`int32`),r=B(r,`int32`)),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Tn,i)}const $d=/* @__PURE__ */ L({maximum_:Qd});
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
function ef(e,t=null,n=!1){let r={x:I(e,`x`,`mean`)},i={axis:t,keepDims:n};return P.runKernel(jn,r,i)}const tf=/* @__PURE__ */ L({mean_:ef});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function nf(e,t=`float32`){if(Ae(e),t===`complex64`){let t=nf(e,`float32`),n=nf(e,`float32`);return za(t,n)}let n=Oe(w(e),t);return P.makeTensor(n,e,t)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function rf(e,t=`float32`){if(Ae(e),t===`complex64`){let t=rf(e,`float32`),n=nf(e,`float32`);return za(t,n)}let n=De(w(e),t);return P.makeTensor(n,e,t)}
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
function af(e,t,{indexing:n=`xy`}={}){if(n!==`xy`&&n!==`ij`)throw TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let r=I(e,`x`,`meshgrid`,e instanceof M?e.dtype:`float32`);if(t===void 0)return[r];let i=I(t,`y`,`meshgrid`,t instanceof M?t.dtype:`float32`),a=w(r.shape),o=w(i.shape);return n===`xy`?(r=G(r,[1,-1]),i=G(i,[-1,1]),[q(rf([o,1],r.dtype),r),q(i,rf([1,a],i.dtype))]):(r=G(r,[-1,1]),i=G(i,[1,-1]),[q(r,rf([1,o],r.dtype)),q(rf([a,1],i.dtype),i)])}
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
function of(e,t){let n=I(e,`a`,`minimum`),r=I(t,`b`,`minimum`);[n,r]=N(n,r),n.dtype===`bool`&&(n=B(n,`int32`),r=B(r,`int32`)),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Nn,i)}const sf=/* @__PURE__ */ L({minimum_:of});
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
function cf(e,t,n){x(n===`reflect`||n===`symmetric`,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);let r=I(e,`x`,`mirrorPad`);if(r.rank===0)throw Error(`mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad`);x(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);let i=+(n===`reflect`);for(let e=0;e<r.rank;e++)x(t[e].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),x(t[e][0]>=0&&t[e][0]<=r.shape[e]-i&&t[e][1]>=0&&t[e][1]<=r.shape[e]-i,()=>`Padding in dimension ${e} cannot be greater than or equal to ${r.shape[e]-i} or less than 0 for input of shape ${r.shape}`);let a={paddings:t,mode:n},o={x:r};return P.runKernel(Pn,o,a)}const lf=/* @__PURE__ */ L({mirrorPad_:cf});
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
function uf(e,t){let n=I(e,`a`,`mod`),r=I(t,`b`,`mod`);[n,r]=N(n,r);let i={a:n,b:r};return P.runKernel(`Mod`,i)}const df=/* @__PURE__ */ L({mod_:uf});
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
function ff(e,t=null,n=!1){e=I(e,`x`,`moments`);let r=ce(t,e.shape),i=tf(e,r,n),a=i.shape;n||(a=uu(i.shape,r));let o=wu(Q(B(e,`float32`),G(i,a)));return{mean:i,variance:tf(o,r,n)}}const pf=/* @__PURE__ */ L({moments_:ff});function mf(e,t,n,r){let i=I(t,`data`,`multiRNNCell`),a=Ia(n,`c`,`multiRNNCell`),o=Ia(r,`h`,`multiRNNCell`),s=i,c=[];for(let t=0;t<e.length;t++){let n=e[t](s,a[t],o[t]);c.push(n[0]),c.push(n[1]),s=n[1]}let l=[],u=[];for(let e=0;e<c.length;e+=2)l.push(c[e]),u.push(c[e+1]);return[l,u]}const hf=/* @__PURE__ */ L({multiRNNCell_:mf});
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
function gf(e,t,n,r=!1){let i=I(e,`logits`,`multinomial`),a=i.size,o=i.rank;if(a<2)throw Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n||=Math.random();let s={logits:o===1?G(i,[1,-1]):i},c={numSamples:t,seed:n,normalized:r},l=P.runKernel(In,s,c);return o===1?G(l,[l.size]):l}const _f=/* @__PURE__ */ L({multinomial_:gf});
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
function vf(e,t){let n=I(e,`a`,`notEqual`,`string_or_numeric`),r=I(t,`b`,`notEqual`,`string_or_numeric`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(zn,i)}const yf=/* @__PURE__ */ L({notEqual_:vf});
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
function bf(e,t,n=1,r=0,i=`int32`){if(t<2)throw Error(`Error in oneHot: depth must be >=2, but it is ${t}`);let a={indices:I(e,`indices`,`oneHot`,`int32`)},o={dtype:i,depth:t,onValue:n,offValue:r};return P.runKernel(Wn,a,o)}const xf=/* @__PURE__ */ L({oneHot_:bf});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Sf(e){let t={x:I(e,`x`,`onesLike`)};return P.runKernel(Un,t)}const Cf=/* @__PURE__ */ L({onesLike_:Sf});function wf(e,t){let n=I(e,`v1`,`outerProduct`),r=I(t,`v2`,`outerProduct`);x(n.rank===1&&r.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`);let i=G(n,[-1,1]),a=G(r,[1,-1]);return q(i,a)}const Tf=/* @__PURE__ */ L({outerProduct_:wf});
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
function Ef(e,t,n=0){let r=I(e,`x`,`pad`);if(r.rank===0)throw Error(`pad(scalar) is not defined. Pass non-scalar to pad`);let i={paddings:t,constantValue:n},a={x:r};return P.runKernel(Kn,a,i)}const Df=/* @__PURE__ */ L({pad_:Ef});function Of(e,t,n=0){return x(t.length===2,()=>`Invalid number of paddings. Must be length of 2.`),Df(e,[t],n)}const kf=/* @__PURE__ */ L({pad1d_:Of});function Af(e,t,n=0){return x(t.length===2&&t[0].length===2&&t[1].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),Df(e,t,n)}const jf=/* @__PURE__ */ L({pad2d_:Af});function Mf(e,t,n=0){return x(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),Df(e,t,n)}const Nf=/* @__PURE__ */ L({pad3d_:Mf});function Pf(e,t,n=0){return x(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),Df(e,t,n)}const Ff=/* @__PURE__ */ L({pad4d_:Pf});
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
function If(e,t,n){let r=I(e,`x`,`spaceToBatchND`);x(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),x(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),x(r.shape.reduce((e,r,i)=>i>0&&i<=t.length?e&&(r+n[i-1][0]+n[i-1][1])%t[i-1]===0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);let i={x:r},a={blockShape:t,paddings:n};return P.runKernel(Tr,i,a)}const Lf=/* @__PURE__ */ L({spaceToBatchND_:If});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Rf(e,t,n,r,i,a,o){i??=[1,1],a??=1,r===0&&(r=`valid`);let s=I(e,`x`,`maxPool`),c=s,l=!1;s.rank===3&&(l=!0,c=G(s,[1,s.shape[0],s.shape[1],s.shape[2]])),x(fc(a,i),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${i}'`);let u=Qs(c.shape,t,a,i,r),d=[u.dilationHeight,u.dilationWidth],f;f=r===`same`?Bf([u.filterHeight,u.filterWidth],d):[[0,0],[0,0]];let p=d[0]===1&&d[1]===1,[m,h]=zf([u.inHeight,u.inWidth],d,f),g=p?r:`valid`,_=p?c:Lf(c,d,m),v=(n===`avg`?()=>vc(_,t,a,g,o):()=>qd(_,t,a,g,o))(),y=p?v:jc(v,d,h);return l?G(y,[y.shape[1],y.shape[2],y.shape[3]]):y}function zf(e,t,n){let r=n.map(e=>e[0]),i=n.map(e=>e[1]),a=e.concat(r,i),o=t.map((e,t)=>(e-a[t]%e)%e),s=i.map((e,t)=>e+o[t]);return[t.map((e,t)=>[r[t],s[t]]),t.map((e,t)=>[0,o[t]])]}function Bf(e,t){let n=e.map((e,n)=>e+(e-1)*(t[n]-1)).map(e=>e-1),r=n.map(e=>Math.floor(e/2)),i=n.map((e,t)=>e-r[t]);return n.map((e,t)=>[r[t],i[t]])}const Vf=/* @__PURE__ */ L({pool_:Rf});
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
function Hf(e,t){let n={x:I(e,`x`,`prelu`),alpha:I(t,`alpha`,`prelu`)};return P.runKernel(Jn,n)}const Uf=/* @__PURE__ */ L({prelu_:Hf});
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
function Wf(e,t=null,n=!1){let r=I(e,`x`,`prod`);r.dtype===`bool`&&(r=B(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return P.runKernel(Yn,i,a)}const Gf=/* @__PURE__ */ L({prod_:Wf});
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
function Kf(e,t,n,r){let i={paramsNestedSplits:e.map((e,t)=>I(e,`tensors${t}`,`raggedGather`,`int32`)),paramsDenseValues:I(t,`paramsDenseValues`,`raggedGather`),indices:I(n,`indices`,`raggedGather`,`int32`)},a={outputRaggedRank:r},o=P.runKernel(Xn,i,a);return{outputNestedSplits:o.slice(0,o.length-1),outputDenseValues:o[o.length-1]}}const qf=/* @__PURE__ */ L({raggedGather_:Kf});
/**
* @license
* Copyright 2022 Google LLC.
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
function Jf(e,t,n){let r=I(e,`starts`,`raggedRange`),i={starts:r,limits:I(t,`limits`,`raggedRange`,r.dtype),deltas:I(n,`deltas`,`raggedRange`,r.dtype)},a=P.runKernel(Zn,i);return{rtNestedSplits:a[0],rtDenseValues:a[1]}}const Yf=/* @__PURE__ */ L({raggedRange_:Jf});
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
function Xf(e,t,n,r,i){let a=I(e,`shape`,`raggedTensorToTensor`,`int32`),o=I(t,`values`,`raggedTensorToTensor`),s={shape:a,values:o,defaultValue:I(n,`defaultValue`,`raggedTensorToTensor`,o.dtype),rowPartitionTensors:r.map((e,t)=>I(e,`tensors${t}`,`raggedTensorToTensor`,`int32`))},c={rowPartitionTypes:i};return P.runKernel(Qn,s,c)}const Zf=/* @__PURE__ */ L({raggedTensorToTensor_:Xf});
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
function Qf(e,t,n){Ae(e);let r=w(e),i=null;if(n==null||n===`float32`)i=new Float32Array(r);else if(n===`int32`)i=new Int32Array(r);else if(n===`bool`)i=new Uint8Array(r);else throw Error(`Unknown data type ${n}`);for(let e=0;e<r;e++)i[e]=t();return P.makeTensor(i,e,n)}const $f=/* @__PURE__ */ L({rand_:Qf});var ep=/* @__PURE__ */ o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=o();t.next=function(){var e=2091639*t.s0+t.c*23283064365386963e-26;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=e|0)},t.c=1,t.s0=n(` `),t.s1=n(` `),t.s2=n(` `),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function i(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function a(e,t){var n=new r(e),a=t&&t.state,o=n.next;return o.int32=function(){return n.next()*4294967296|0},o.double=function(){return o()+(o()*2097152|0)*11102230246251565e-32},o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}function o(){var e=4022871197;return function(t){t=String(t);for(var n=0;n<t.length;n++){e+=t.charCodeAt(n);var r=.02519603282416938*e;e=r>>>0,r-=e,r*=e,e=r>>>0,r-=e,e+=r*4294967296}return(e>>>0)*23283064365386963e-26}}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.alea=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),tp=/* @__PURE__ */ o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor128=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),np=/* @__PURE__ */ o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^(e^e<<1))|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorwow=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),rp=/* @__PURE__ */ o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.x,n=t.i,r=e[n],i;return r^=r>>>7,i=r^r<<24,r=e[n+1&7],i^=r^r>>>10,r=e[n+3&7],i^=r^r>>>3,r=e[n+4&7],i^=r^r<<7,r=e[n+7&7],r^=r<<13,i^=r^r<<9,e[n]=i,t.i=n+1&7,i};function n(e,t){var n,r=[];if(t===(t|0))r[0]=t;else for(t=``+t,n=0;n<t.length;++n)r[n&7]=r[n&7]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&r[n]===0;++n);for(n==8?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}n(t,e)}function i(e,t){return t.x=e.x.slice(),t.i=e.i,t}function a(e,t){e??=+/* @__PURE__ */ new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.x&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorshift7=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),ip=/* @__PURE__ */ o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.w,n=t.X,r=t.i,i,a;return t.w=e=e+1640531527|0,a=n[r+34&127],i=n[r=r+1&127],a^=a<<13,i^=i<<17,a^=a>>>15,i^=i>>>12,a=n[r]=a^i,t.i=r,a+(e^e>>>16)|0};function n(e,t){var n,r,i,a,o,s=[],c=128;for(t===(t|0)?(r=t,t=null):(t+=`\0`,r=0,c=Math.max(c,t.length)),i=0,a=-32;a<c;++a)t&&(r^=t.charCodeAt((a+32)%t.length)),a===0&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,a>=0&&(o=o+1640531527|0,n=s[a&127]^=r+o,i=n==0?i+1:0);for(i>=128&&(s[(t&&t.length||0)&127]=-1),i=127,a=512;a>0;--a)r=s[i+34&127],n=s[i=i+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,s[i]=r^n;e.w=o,e.X=s,e.i=i}n(t,e)}function i(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function a(e,t){e??=+/* @__PURE__ */ new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.X&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor4096=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),ap=/* @__PURE__ */ o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.b,n=t.c,r=t.d,i=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^i,i=i-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^i,t.a=i-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=e|0):n+=e;for(var r=0;r<n.length+20;r++)t.b^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.tychei=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),op=/* @__PURE__ */ o(((e,t)=>{(function(e,n,r){var i=256,a=6,o=52,s=`random`,c=r.pow(i,a),l=r.pow(2,o),u=l*2,d=i-1,f;function p(e,t,o){var d=[];t=t==1?{entropy:!0}:t||{};var f=_(g(t.entropy?[e,y(n)]:e??v(),3),d),p=new m(d),b=function(){for(var e=p.g(a),t=c,n=0;e<l;)e=(e+n)*i,t*=i,n=p.g(1);for(;e>=u;)e/=2,t/=2,n>>>=1;return(e+n)/t};return b.int32=function(){return p.g(4)|0},b.quick=function(){return p.g(4)/4294967296},b.double=b,_(y(p.S),n),(t.pass||o||function(e,t,n,i){return i&&(i.S&&h(i,p),e.state=function(){return h(p,{})}),n?(r[s]=e,t):e})(b,f,`global`in t?t.global:this==r,t.state)}function m(e){var t,n=e.length,r=this,a=0,o=r.i=r.j=0,s=r.S=[];for(n||(e=[n++]);a<i;)s[a]=a++;for(a=0;a<i;a++)s[a]=s[o=d&o+e[a%n]+(t=s[a])],s[o]=t;(r.g=function(e){for(var t,n=0,a=r.i,o=r.j,s=r.S;e--;)t=s[a=d&a+1],n=n*i+s[d&(s[a]=s[o=d&o+t])+(s[o]=t)];return r.i=a,r.j=o,n})(i)}function h(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function g(e,t){var n=[],r=typeof e,i;if(t&&r==`object`)for(i in e)try{n.push(g(e[i],t-1))}catch{}return n.length?n:r==`string`?e:e+`\0`}function _(e,t){for(var n=e+``,r,i=0;i<n.length;)t[d&i]=d&(r^=t[d&i]*19)+n.charCodeAt(i++);return y(t)}function v(){try{var t;return f&&(t=f.randomBytes)?t=t(i):(t=new Uint8Array(i),(e.crypto||e.msCrypto).getRandomValues(t)),y(t)}catch{var r=e.navigator,a=r&&r.plugins;return[+/* @__PURE__ */ new Date,e,a,e.screen,y(n)]}}function y(e){return String.fromCharCode.apply(0,e)}if(_(r.random(),n),typeof t==`object`&&t.exports){t.exports=p;try{f=fs()}catch{}}else typeof define==`function`&&define.amd?define(function(){return p}):r[`seed`+s]=p})(typeof self<`u`?self:e,[],Math)})),sp=/* @__PURE__ */ l((/* @__PURE__ */ o(((e,t)=>{var n=ep(),r=tp(),i=np(),a=rp(),o=ip(),s=ap(),c=op();c.alea=n,c.xor128=r,c.xorwow=i,c.xorshift7=a,c.xor4096=o,c.tychei=s,t.exports=c})))()),cp=/* @__PURE__ */ s({TEST_EPSILON_FLOAT16:()=>lp,createVideoElement:()=>bp,encodeStrings:()=>yp,expectArrayBuffersEqual:()=>vp,expectArraysClose:()=>up,expectArraysEqual:()=>mp,expectNumbersClose:()=>hp,expectPromiseToFail:()=>pp,expectValuesInRange:()=>_p,play:()=>xp,testEpsilon:()=>dp})
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
;const lp=.1;function up(e,t,n){return n??=dp(),fp(e,t,(e,t)=>gp(e,t,n))}function dp(){return P.backend.floatPrecision()===32?.001:lp}function fp(e,t,n){let r=!0;if((j(e)||j(t))&&(r=!1),j(e)&&j(t)&&(r=!0),r){let n=e.constructor.name,r=t.constructor.name;if(n!==r)throw Error(`Arrays are of different type. Actual: ${n}. Expected: ${r}`)}if(Array.isArray(e)&&Array.isArray(t)){let n=Na(e),r=Na(t);if(!E(n,r))throw Error(`Arrays have different shapes. Actual: [${n}]. Expected: [${r}]`)}let i=j(e)?e:Bi(e),a=j(t)?t:Bi(t);if(i.length!==a.length)throw Error(`Arrays have different lengths actual: ${i.length} vs expected: ${a.length}.\nActual:   ${i}.\nExpected: ${a}.`);for(let e=0;e<a.length;++e){let t=i[e],r=a[e];if(!n(t,r))throw Error(`Arrays differ: actual[${e}] = ${t}, expected[${e}] = ${r}.\nActual:   ${i}.\nExpected: ${a}.`)}typeof expect<`u`&&expect().nothing()}function pp(e,t){e().then(()=>t.fail(),()=>t()),typeof expect<`u`&&expect().nothing()}function mp(e,t){let n=typeof t==`string`||typeof t==`number`||typeof t==`boolean`?[t]:t;return _e(e)||_e(e[0])||_e(t)||_e(t[0])?fp(e,n,(e,t)=>e==t):fp(e,t,(e,t)=>gp(e,t,0))}function hp(e,t,n){if(n??=dp(),!gp(e,t,n))throw Error(`Numbers differ: actual === ${e}, expected === ${t}`);typeof expect<`u`&&expect().nothing()}function gp(e,t,n){return!isFinite(e)&&!isFinite(t)||!(isNaN(e)||isNaN(t)||Math.abs(e-t)>n)}function _p(e,t,n){for(let r=0;r<e.length;r++)if(e[r]<t||e[r]>n)throw Error(`Value out of range:${e[r]} low: ${t}, high: ${n}`)}function vp(e,t){let n=new Float32Array(e),r=new Float32Array(t);if(n.length!==r.length)throw Error(`Expected ArrayBuffer to be of length ${r.length}, but it was ${n.length}`);for(let e=0;e<r.length;e++)if(n[e]!==r[e])throw Error(`Expected ArrayBuffer value at ${e} to be ${r[e]} but got ${n[e]} instead`)}function yp(e){for(let t=0;t<e.length;t++){let n=e[t];Array.isArray(n)?yp(n):e[t]=Ri(n)}return e}function bp(e){let t=document.createElement(`video`);return`playsInline`in t&&(t.playsInline=!0),t.muted=!0,t.loop=!0,t.style.position=`fixed`,t.style.left=`0px`,t.style.top=`0px`,t.preload=`auto`,t.appendChild(e),new Promise(e=>{t.addEventListener(`loadeddata`,n=>e(t)),t.load()})}async function xp(e){await e.play(),`requestVideoFrameCallback`in e&&await new Promise(t=>{e.requestVideoFrameCallback(t)})}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Sp=class{constructor(e,t,n,r,i){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);let a=i||Math.random();this.random=sp.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){let e=this.nextVal;return this.nextVal=NaN,e}let e,t,n=!1;for(;!n;){let r,i,a;do r=2*this.random()-1,i=2*this.random()-1,a=r*r+i*i;while(a>=1||a===0);let o=Math.sqrt(-2*Math.log(a)/a);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*i*o,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype===`float32`?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}},Cp=class{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;let i=r||Math.random();this.randu=sp.alea(i.toString()),this.randn=new Sp(0,1,n,!1,this.randu()),this.d=e<1?e+2/3:e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,i,a;for(;;){do r=this.randn.nextValue(),a=1+this.c*r;while(a<=0);if(a*=a*a,e=r*r,t=1-.331*e*e,n=.5*e+this.d*(1-a+Math.log(a)),i=this.randu(),i<t||Math.log(i)<n)break}return a=1/this.beta*this.d*a,this.alpha<1&&(a*=this.randu()**(1/this.alpha)),this.convertValue(a)}convertValue(e){return this.dtype===`float32`?e:Math.round(e)}},wp=class{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype===`float32`,this.min=e,this.range=t-e,this.dtype=n,r??=Math.random(),typeof r==`number`&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=sp.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}};
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
function Tp(e,t,n=1,r=`float32`,i){if(Ae(e),n??=1,r??=`float32`,r!==`float32`&&r!==`int32`)throw Error(`Unsupported data type ${r}`);let a=new Cp(t,n,r,i),o=gs(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}const Ep=/* @__PURE__ */ L({randomGamma_:Tp});
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
function Dp(e,t=0,n=1,r,i){if(Ae(e),r!=null&&r===`bool`)throw Error(`Unsupported data type ${r}`);let a=new Sp(t,n,r,!1,i),o=gs(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}const Op=/* @__PURE__ */ L({randomNormal_:Dp});
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
function kp(e,t,n){if(t!=null&&t===`bool`)throw Error(`Unsupported data type ${t}`);return Op(e,0,1,t,n)}const Ap=/* @__PURE__ */ L({randomStandardNormal_:kp});
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
function jp(e,t=0,n=1,r=`float32`,i){Ae(e);let a=gs(e,r),o=new wp(t,n,null,i);for(let e=0;e<a.values.length;e++)a.values[e]=o.nextValue();return a.toTensor()}const Mp=/* @__PURE__ */ L({randomUniform_:jp});
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
function Np(e,t,n,r){return Mp(e,t,n,`int32`,r)}const Pp=/* @__PURE__ */ L({randomUniformInt_:Np});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Fp(e,t,n=1,r=`float32`){if(n===0)throw Error(`Cannot have a step of zero`);let i={start:e,stop:t,step:n,dtype:r};return P.runKernel($n,{},i)}
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
function Ip(e){let t={input:I(e,`input`,`real`)};return P.runKernel(er,t)}const Lp=/* @__PURE__ */ L({real_:Ip});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Rp(e){let t={x:I(e,`x`,`reciprocal`)};return P.runKernel(tr,t)}const zp=/* @__PURE__ */ L({reciprocal_:Rp});
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
function Bp(e){let t={x:I(e,`x`,`relu`)};return P.runKernel(nr,t)}const Vp=/* @__PURE__ */ L({relu_:Bp});
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
function Hp(e){let t={x:I(e,`x`,`relu6`)};return P.runKernel(cr,t)}const Up=/* @__PURE__ */ L({relu6_:Hp});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Wp(e,t){let n={x:I(e,`x`,`reverse`)},r={dims:t};return P.runKernel(lr,n,r)}const Gp=/* @__PURE__ */ L({reverse_:Wp});
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
function Kp(e){let t=I(e,`x`,`reverse`);return x(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),Gp(t,0)}const qp=/* @__PURE__ */ L({reverse1d_:Kp});
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
function Jp(e,t){let n=I(e,`x`,`reverse`);return x(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),Gp(n,t)}const Yp=/* @__PURE__ */ L({reverse2d_:Jp});
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
function Xp(e,t){let n=I(e,`x`,`reverse`);return x(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),Gp(n,t)}const Zp=/* @__PURE__ */ L({reverse3d_:Xp});
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
function Qp(e,t){let n=I(e,`x`,`reverse`);return x(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),Gp(n,t)}const $p=/* @__PURE__ */ L({reverse4d_:Qp});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function em(e){let t={x:I(e,`x`,`round`)};return P.runKernel(ur,t)}const tm=/* @__PURE__ */ L({round_:em});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function nm(e){let t={x:I(e,`x`,`rsqrt`,`float32`)};return P.runKernel(dr,t)}const rm=/* @__PURE__ */ L({rsqrt_:nm});
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
function im(e){let t={x:I(e,`x`,`selu`)};return P.runKernel(gr,t)}const am=/* @__PURE__ */ L({selu_:im});function om(e,t,n,r,i,a=[1,1],o=`NHWC`){let s=I(e,`x`,`separableConv2d`),c=I(t,`depthwiseFilter`,`separableConv2d`),l=I(n,`pointwiseFilter`,`separableConv2d`),u=s,d=!1;if(s.rank===3&&(d=!0,u=G(s,[1,s.shape[0],s.shape[1],s.shape[2]])),o===`NCHW`)throw Error(`separableConv2d currently does not support dataFormat NCHW; only NHWC is supported`);x(u.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${u.rank}.`),x(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),x(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),x(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),x(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);let f=c.shape[2],p=c.shape[3];x(l.shape[2]===f*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*p}, but got ${l.shape[2]}.`);let m=Fl(u,c,r,i,o,a),h=ll(m,l,1,`valid`,o);return d?G(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const sm=/* @__PURE__ */ L({separableConv2d_:om});
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
async function cm(e,t){let n=I(e,`x`,`setdiff1d`),r=I(t,`y`,`setdiff1d`);x(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),x(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),x(r.rank===1,()=>`y should be 1D tensor, but got y (${r.shape}).`);let i=await n.data(),a=await r.data(),o=new Set(a),s=0;for(let e=0;e<i.length;e++)o.has(i[e])||s++;let c=new Qi([s],n.dtype),l=new Qi([s],`int32`);for(let e=0,t=0;e<i.length;e++)o.has(i[e])||(c.values[t]=i[e],l.values[t]=e,t++);return[c.toTensor(),l.toTensor()]}const lm=cm;
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function um(e){let t={x:I(e,`x`,`sign`)};return P.runKernel(br,t)}const dm=/* @__PURE__ */ L({sign_:um});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function fm(e){let t={x:I(e,`x`,`sin`,`float32`)};return P.runKernel(`Sin`,t)}const pm=/* @__PURE__ */ L({sin_:fm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function mm(e){let t={x:I(e,`x`,`sinh`)};return P.runKernel(yr,t)}const hm=/* @__PURE__ */ L({sinh_:mm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function gm(e,t,n){let r=I(e,`x`,`slice1d`);return x(r.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),J(r,[t],[n])}const _m=/* @__PURE__ */ L({slice1d_:gm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function vm(e,t,n){let r=I(e,`x`,`slice2d`);return x(r.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),J(r,t,n)}const ym=/* @__PURE__ */ L({slice2d_:vm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function bm(e,t,n){let r=I(e,`x`,`slice3d`);return x(r.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),J(r,t,n)}const xm=/* @__PURE__ */ L({slice3d_:bm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Sm(e,t,n){let r=I(e,`x`,`slice4d`);return x(r.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),J(r,t,n)}const Cm=/* @__PURE__ */ L({slice4d_:Sm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function wm(e,t=-1){let n=I(e,`logits`,`softmax`,`float32`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);let r={logits:n},i={dim:t};return P.runKernel(Dr,r,i)}const Tm=/* @__PURE__ */ L({softmax_:wm});
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
function Em(e){x(e.dtype===`complex64`,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);let t={input:e};return P.runKernel(`FFT`,t)}const Dm=/* @__PURE__ */ L({fft_:Em});
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
function Om(e){x(e.dtype===`complex64`,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);let t={input:e};return P.runKernel(sn,t)}const km=/* @__PURE__ */ L({ifft_:Om});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Am(e){let t=e.shape[e.shape.length-1],n=e.size/t,r;if(t<=2){let i=G(e,[n,t]);r=km(i)}else{let i=[n,2*(t-1)],a=G(Lp(e),[n,t]),o=G(Xu(e),[n,t]),s=Gp(J(a,[0,1],[n,t-2]),1),c=U(Gp(J(o,[0,1],[n,t-2]),1),X(-1)),l=K([a,s],1),u=K([o,c],1),d=G(za(l,u),[i[0],i[1]]);r=km(d)}if(r=Lp(r),e.rank===3&&e.shape[0]!==0){let t=r,n=e.shape[0];r=G(r,[n,r.shape[0]/n,r.shape[1]]),t.dispose()}return r}const jm=/* @__PURE__ */ L({irfft_:Am});
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
function Mm(e,t,n=0){let r={x:I(e,`x`,`split`)},i={numOrSizeSplits:t,axis:n};return P.runKernel(Er,r,i)}const Nm=/* @__PURE__ */ L({split_:Mm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Pm(e,t){x(e.dtype===`float32`,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1],r=e.size/n,i;if(t!=null&&t<n){let r=e.shape.map(e=>0),a=e.shape.map(e=>e);a[e.shape.length-1]=t,i=J(e,r,a),n=t}else if(t!=null&&t>n){let r=e.shape.map(e=>e);r[e.shape.length-1]=t-n,i=K([e,nf(r)],e.shape.length-1),n=t}else i=e;let a=Jl(i),o=G(za(i,a),[r,n]),s=Dm(o),c=Math.floor(n/2)+1,l=Lp(s),u=Xu(s),d=Nm(l,[c,n-c],l.shape.length-1),f=Nm(u,[c,n-c],u.shape.length-1),p=i.shape.slice();return p[i.shape.length-1]=c,G(za(d[0],f[0]),p)}const Fm=/* @__PURE__ */ L({rfft_:Pm});
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
function Im(e,t){let n=I(e,`a`,`squaredDifference`),r=I(t,`b`,`squaredDifference`);[n,r]=N(n,r),Y(n.shape,r.shape);let i={a:n,b:r};return P.runKernel(Nr,i,{})}const Lm=/* @__PURE__ */ L({squaredDifference_:Im});
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
function Rm(e,t){let n=I(e,`x`,`squeeze`,`string_or_numeric`);return G(n,le(n.shape,t).newShape)}const zm=/* @__PURE__ */ L({squeeze_:Rm});
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
function Bm(e,t=0){let n=Ia(e,`tensors`,`stack`,`string_or_numeric`);x(n.length>=1,()=>`Pass at least one tensor to tf.stack`),n.length>0&&x(t<=n[0].rank,()=>`Axis must be <= rank of the tensor`);let r=n,i={axis:t};return P.runKernel(Gn,r,i)}const Vm=/* @__PURE__ */ L({stack_:Bm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Hm(e,t=0){let n={x:I(e,`x`,`step`)},r={alpha:t};return P.runKernel(Zr,n,r)}const Um=/* @__PURE__ */ L({step_:Hm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Wm(e,t,n,r,i=0,a=0,o=0,s=0,c=0){let l={x:I(e,`x`,`stridedSlice`,`string_or_numeric`)},u={begin:t,end:n,strides:r,beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};return P.runKernel(Ir,l,u)}const Gm=/* @__PURE__ */ L({stridedSlice_:Wm});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Km(e){let t={x:I(e,`x`,`tan`,`float32`)};return P.runKernel(`Tan`,t)}const qm=/* @__PURE__ */ L({tan_:Km});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Jm(e,t){C(e);let n=Na(e,t);if(n.length!==1)throw Error(`tensor1d() requires values to be a flat/TypedArray`);return Ba(e,null,n,t)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ym(e,t,n){if(C(e),t!=null&&t.length!==2)throw Error(`tensor2d() requires shape to have two numbers`);let r=Na(e,n);if(r.length!==2&&r.length!==1)throw Error(`tensor2d() requires values to be number[][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Ba(e,t,r,n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Xm(e,t,n){if(C(e),t!=null&&t.length!==3)throw Error(`tensor3d() requires shape to have three numbers`);let r=Na(e,n);if(r.length!==3&&r.length!==1)throw Error(`tensor3d() requires values to be number[][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor3d() requires shape to be provided when `values` are a flat array");return Ba(e,t,r,n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Zm(e,t,n){if(C(e),t!=null&&t.length!==4)throw Error(`tensor4d() requires shape to have four numbers`);let r=Na(e,n);if(r.length!==4&&r.length!==1)throw Error(`tensor4d() requires values to be number[][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor4d() requires shape to be provided when `values` are a flat array");return Ba(e,t,r,n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Qm(e,t,n){if(C(e),t!=null&&t.length!==5)throw Error(`tensor5d() requires shape to have five numbers`);let r=Na(e,n);if(r.length!==5&&r.length!==1)throw Error(`tensor5d() requires values to be number[][][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor5d() requires shape to be provided when `values` are a flat array");return Ba(e,t,r,n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function $m(e,t,n){if(C(e),t!=null&&t.length!==6)throw Error(`tensor6d() requires shape to have six numbers`);let r=Na(e,n);if(r.length!==6&&r.length!==1)throw Error(`tensor6d() requires values to be number[][][][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor6d() requires shape to be provided when `values` are a flat array");return t||=r,Ba(e,t,r,n)}var eh=/* @__PURE__ */ s({calculateShapes:()=>rh,validateInput:()=>nh,validateUpdateShape:()=>th});function th(e,t,n){let r=t.rank>1?t.shape[t.rank-1]:1,i=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${i}.`;if(n.rank<i)throw Error(a+` update.rank < ${i}. `);if(e.length<r+(n.rank-i))throw Error(a+` Output shape length < ${r+(n.rank-i)}`);if(n.rank!==i+e.length-r)throw Error(a+` update.rank != ${i+e.length-r}`);for(let e=0;e<i;++e)if(n.shape[e]!==t.shape[e])throw Error(a+` updates.shape[${e}] (${n.shape[e]}) != indices.shape[${e}] (${t.shape[e]}).`);for(let t=0;t<n.rank-i;++t)if(n.shape[t+i]!==e[t+r])throw Error(a+` updates.shape[${t+i}] (${n.shape[t+i]}) != shape[${t+i}] (${e[t+i]})`)}function nh(e,t,n){if(t.rank<1)throw Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!==`int32`)throw Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw Error(`Updates specified for empty output. updates shape: ${e.shape}`)}th(n,t,e)}function rh(e,t,n){let r=t.shape.length,i=r>1?t.shape[r-1]:1,a=n.length,o=1;for(let e=i;e<a;++e)o*=n[e];let s=i<1?1:i,c=w(t.shape)/s,l=[...Ce(n.slice(0,i)),1],u=w(n);return{sliceRank:i,numUpdates:c,sliceSize:o,strides:l,outputSize:u}}
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
function ih(e,t,n){let r=I(e,`tensor`,`tensorScatterupdate`),i=I(t,`indices`,`tensorScatterupdate`,`int32`),a=I(n,`updates`,`tensorScatterupdate`);if(nh(a,i,r.shape),r.dtype!==a.dtype)throw Error(`tensor and updates must have the same dtype, instead they are ${r.dtype} and ${a.dtype}.`);let o={tensor:r,indices:i,updates:a};return P.runKernel(pr,o,{})}const ah=L({tensorScatterUpdate_:ih});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function oh(e,t=1,n=!0){let r=I(e,`x`,`topk`);if(r.rank===0)throw Error(`topk() expects the input to be of rank 1 or higher`);let i=r.shape[r.shape.length-1];if(t<0)throw Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>i)throw Error(`'k' passed to topk() must be <= the last dimension (${i}) but got ${t}`);let a={x:r},o={k:t,sorted:n},[s,c]=P.runKernel(Wr,a,o);return{values:s,indices:c}}const sh=/* @__PURE__ */ L({topk_:oh});
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
function ch(e,t=0,n=1,r,i){if(Ae(e),r!=null&&r===`bool`)throw Error(`Unsupported data type $ { dtype }`);let a=new Sp(t,n,r,!0,i),o=gs(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}const lh=/* @__PURE__ */ L({truncatedNormal_:ch});
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
function uh(e,t=0){let n=I(e,`x`,`unique`,`string_or_numeric`);x(n.rank>0,()=>`The input tensor must be at least 1D`);let r={x:n},i={axis:t},[a,o]=P.runKernel(qr,r,i);return{values:a,indices:o}}const dh=/* @__PURE__ */ L({unique_:uh});
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
function fh(e,t,n){let r=I(e,`x`,`unsortedSegmentSum`),i=I(t,`segmentIds`,`unsortedSegmentSum`,`int32`);x(te(n),()=>`numSegments must be of dtype int`);let a={x:r,segmentIds:i},o={numSegments:n};return P.runKernel(Yr,a,o)}const ph=/* @__PURE__ */ L({unsortedSegmentSum_:fh});
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
function mh(e,t=0){let n=I(e,`x`,`unstack`,`string_or_numeric`);x(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);let r={value:n},i={axis:t};return P.runKernel(Jr,r,i)}const hh=/* @__PURE__ */ L({unstack_:mh});
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
function gh(e,t){return Wd(e,t,`right`)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function _h(e,t=!0,n,r){return P.makeVariable(e,t,n,r)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function vh(e,t){let n=[];for(let e=0;e<t.length;e++)t[e]&&n.push(e);let r=gs(e,`int32`),i=gs([n.length,e.length],`int32`);for(let t=0;t<n.length;t++){let a=r.indexToLoc(n[t]),o=t*e.length;i.values.set(a,o)}return i.toTensor()}
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
async function yh(e){let t=I(e,`condition`,`whereAsync`,`bool`),n=await t.data(),r=vh(t.shape,n);return e!==t&&t.dispose(),r}const bh=yh;
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
async function xh(e,t,n){let r=I(e,`tensor`,`boolMask`),i=I(t,`mask`,`boolMask`,`bool`),a=n??0,o=i.rank,s=r.shape;x(o>0,()=>`mask cannot be scalar`),S(s.slice(a,a+o),i.shape,`mask's shape must match the first K dimensions of tensor's shape,`);let c=1;for(let e=a;e<a+o;e++)c*=s[e];let l=s.slice(0,a).concat([c],s.slice(a+o)),u=G(r,l),d=G(i,[-1]),f=await bh(d),p=zm(f,[1]),m=Wu(u,p,a);return e!==r&&r.dispose(),t!==i&&i.dispose(),p.dispose(),u.dispose(),d.dispose(),f.dispose(),m}const Sh=xh;
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ch(e,t,n){let r=I(e,`x`,`transpose`);if(t??=r.shape.map((e,t)=>t).reverse(),x(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(e=>{x(e>=0&&e<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();let i={x:r},a={perm:t};return r.dtype===`complex64`?R(()=>{let e=Lp(r),t=Xu(r);return e=P.runKernel(Kr,{x:e},a),t=P.runKernel(Kr,{x:t},a),n&&(t=wd(t)),za(e,t)}):P.runKernel(Kr,i,a)}const wh=/* @__PURE__ */ L({transpose_:Ch});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Th(e,t,n,r,i=!0){let a=I(e,`v`,`movingAverage`),o=I(t,`x`,`movingAverage`),s=I(n,`decay`,`movingAverage`);ga(a,o),x(E(a.shape,o.shape),()=>`Shape mismatch in v and x`);let c=X(1),l=Q(c,s),u=U(Q(o,a),l);if(i){x(r!=null,()=>`When using zeroDebias: true, step is required.`);let e=I(r,`step`,`movingAverage`);u=H(u,Q(c,bu(s,e)))}return V(a,u)}const Eh=/* @__PURE__ */ L({movingAverage_:Th});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Dh(e,t,n){Ae(n);let r=I(e,`indices`,`scatterND`,`int32`),i=I(t,`updates`,`scatterND`);nh(i,r,n);let a={indices:r,updates:i},o={shape:n};return P.runKernel(fr,a,o)}const Oh=/* @__PURE__ */ L({scatterND_:Dh});function kh(e,t,n,r){if(e.dtype!==`int32`)throw Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);let i=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);let o=t.size;if(!(t.rank===0||t.rank===1&&o===i))throw Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${i}]`);if(t.dtype!==r.dtype)throw Error(`sparseValues.dtype must match defaultValues.dtype`)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Ah(e,t,n,r=0){Ae(n);let i=I(e,`sparseIndices`,`sparseToDense`,`int32`),a=I(t,`sparseValues`,`sparseToDense`,`string_or_numeric`),o=I(r,`defaultValue`,`sparseToDense`,a.dtype);kh(i,a,n,o);let s={sparseIndices:i,sparseValues:a,defaultValue:o},c={outputShape:n};return P.runKernel(Mr,s,c)}const jh=/* @__PURE__ */ L({sparseToDense_:Ah});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Mh(e,t){let n=I(t,`indices`,`gatherND`,`int32`),r={params:I(e,`x`,`gatherND`,`string_or_numeric`),indices:n};return P.runKernel(nn,r)}const Nh=/* @__PURE__ */ L({gatherND_:Mh});
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
function Ph(e,t){if(t==null)return e.shape.slice();if(E(e.shape,t))return t;if(e.shape.length===t.length){let n=[];for(let r=0;r<e.shape.length;r++)t[r]==null&&e.shape[r]!=null?n.push(e.shape[r]):n.push(t[r]);return n}return t}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Fh(e,t,n,r){let i=I(e,`x`,`dropout`);if(x(i.dtype===`float32`,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${i.dtype} tensor instead.`),x(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof M?i.clone():i;let a=Ph(i,n),o=1-t,s=H(Hu(V(Mp(a,0,1,`float32`,r),o)),o);return U(i,s)}const Ih=/* @__PURE__ */ L({dropout_:Fh});
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
function Lh(e){return Math.floor(2**Math.ceil(Math.log(e)/Math.log(2)))}function Rh(e,t,n){let r=1-e%2,i=new Float32Array(e);for(let a=0;a<e;++a){let o=2*Math.PI*a/(e+r-1);i[a]=t-n*Math.cos(o)}return Jm(i,`float32`)}
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
async function zh(e,t,n=1){let r=I(e,`predictions`,`inTopK`),i=I(t,`targets`,`inTopK`);x(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),x(r.rank-1===i.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${i.rank}`),S(r.shape.slice(0,r.shape.length-1),i.shape,`predictions's shape should be align with the targets' shape, except the last dimension.`);let a=r.shape[r.shape.length-1];x(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);let o=await r.data(),s=await i.data(),[c,l]=[o.length/a,a],u=ue(`bool`,c);for(let e=0;e<c;e++){let t=e*l,r=o.subarray(t,t+l),i=[];for(let e=0;e<r.length;e++)i.push({value:r[e],index:e});i.sort((e,t)=>t.value-e.value),u[e]=0;for(let t=0;t<n;t++)if(i[t].index===s[e]){u[e]=1;break}}return e!==r&&r.dispose(),t!==i&&i.dispose(),Va(u,i.shape,`bool`)}const Bh=zh;
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
function Vh(e,t,n,r,i,a=`NHWC`,o){let s=e;e.rank===3&&(s=G(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=G(t,[1,t.shape[0],t.shape[1],t.shape[2]])),x(s.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${s.shape}.`),x(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),x(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);let l=a===`NHWC`?s.shape[3]:s.shape[1],u=a===`NHWC`?c.shape[3]:c.shape[1];x(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),x(u===n[3],()=>`Error in conv2dDerFilter: depth of dy (${u}) must match output depth for filter (${n[3]}).`),hc(`conv2dDerFilter`,i,o);let d={x:s,dy:c},f={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,filterShape:n};return P.runKernel(xt,d,f)}const Hh=/* @__PURE__ */ L({conv2DBackpropFilter_:Vh});
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
function Uh(e,t,n){if(n==null||n===`linear`)return e;if(n===`relu`)return U(e,Um(t));throw Error(`Cannot compute gradient for fused activation ${n}.`)}function Wh(e,t){let n=t,r=Hl(e.shape,t.shape);return r.length>0&&(n=Z(n,r)),G(n,e.shape)}function Gh(e,t,n,r){if(t===`linear`)return e;if(t===`relu`)return Vp(e);if(t===`elu`)return nu(e);if(t===`relu6`)return Up(e);if(t===`prelu`)return Uf(e,n);if(t===`leakyrelu`)return id(e,r);if(t===`sigmoid`)return wc(e);throw Error(`Unknown fused activation ${t}.`)}const Kh=(e,t)=>!(e>0)||t===`linear`;
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
function qh({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(c||=`linear`,Kh(P.state.gradientDepth,c)===!1){x(i===`NHWC`,()=>`Error in fused conv2d: got dataFormat of ${i} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let d=ll(e,t,n,r,i,a,o);return s!=null&&(d=V(d,s)),Gh(d,c,l,u)}let d=I(e,`x`,`conv2d`,`float32`),f=I(t,`filter`,`conv2d`,`float32`),p=d,m=!1;d.rank===3&&(m=!0,p=G(d,[1,d.shape[0],d.shape[1],d.shape[2]])),x(p.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${p.rank}.`),x(f.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),hc(`fused conv2d`,r,o);let h=i===`NHWC`?p.shape[3]:p.shape[1];x(f.shape[2]===h,()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${f.shape[2]}.`),x(fc(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let g=ec(p.shape,f.shape,n,a,r,o),_;s!=null&&(_=I(s,`bias`,`fused conv2d`),[_]=N(_,d),i===`NHWC`?Y(g.outShape,_.shape):(x(_.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${_.shape.length}.`),x(_.shape.length===0||_.shape[0]===g.outChannels||_.shape[0]===1,()=>`Error in fused conv2d: bias shape (${_.shape}) is not compatible with the number of output channels (${g.outChannels})`)));let v;if(l!=null){let e=l.shape;if(x(e.length<=1||e.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${e.length}.`),e.length===1)x(e[0]===1||e[0]===g.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the number of output channels (${g.outChannels}).`);else if(e.length===3)try{Y(e,g.outShape)}catch{let t=`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the output shape of the conv2d (${g.outShape}).`;throw Error(t)}v=I(l,`prelu weights`,`fused conv2d`)}let y=(e,t)=>{x(i===`NHWC`,()=>`Error in gradient of fused conv2D: got dataFormat of ${i} but only NHWC is currently supported.`);let[o,s,l,u]=t,d=Uh(e,l,c);x(dc(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);let f=[pl(s.shape,d,o,n,r),Hh(s,d,o.shape,n,r)];if(u!=null){let e=Wh(u,d);f.push(e)}return f},b={x:p,filter:f,bias:_,preluActivationWeights:v},S={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?xd((e,t,n)=>{let r=P.runKernel(ti,b,S);return n([t,e,r]),m&&(r=G(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:y}})(p,f):xd((e,t,n,r)=>{let i=P.runKernel(ti,b,S);return r([t,e,i,n]),m&&(i=G(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:y}})(p,f,_)}const Jh=/* @__PURE__ */ L({fusedConv2d_:qh});
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
function Yh(e,t,n,r,i,a=[1,1],o){let s=e;e.rank===3&&(s=G(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=G(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={x:s,dy:c},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,filterShape:n};return P.runKernel(Pt,l,u)}const Xh=L({depthwiseConv2dNativeBackpropFilter_:Yh});
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
function Zh(e,t,n,r,i,a=[1,1],o){let s=t,c=!1;t.rank===3&&(c=!0,s=G(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={dy:s,filter:n},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,inputShape:e},d=P.runKernel(Ft,l,u);return c?G(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const Qh=L({depthwiseConv2dNativeBackpropInput_:Zh});
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
function $h({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(Kh(P.state.gradientDepth,c)===!1){let d=Fl(e,t,n,r,i,a,o);return s!=null&&(d=V(d,s)),Gh(d,c,l,u)}let d=I(e,`x`,`depthwiseConv2d`,`float32`),f=I(t,`filter`,`depthwiseConv2d`,`float32`),p=d,m=!1;d.rank===3&&(m=!0,p=G(d,[1,d.shape[0],d.shape[1],d.shape[2]])),x(p.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${p.rank}.`),x(f.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${f.rank}.`),x(p.shape[3]===f.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${p.shape[3]}) must match the inChannels dimension in filter ${f.shape[2]}.`),a??=[1,1],x(fc(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),hc(`fused depthwiseConv2d`,r,o);let h=ec(p.shape,f.shape,n,a,r,o,!0),g;s!=null&&(g=I(s,`bias`,`fused conv2d`),[g]=N(g,d),Y(h.outShape,g.shape));let _;l!=null&&(_=I(l,`prelu weights`,`fused depthwiseConv2d`));let v=(e,t)=>{x(dc(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);let[i,s,l,u]=t,d=Uh(e,l,c),f=Qh(s.shape,d,i,n,r,a,o),p=Xh(s,d,i.shape,n,r,a,o);return u==null?[f,p]:[f,p,Wh(g,d)]},y={x:p,filter:f,bias:g,preluActivationWeights:_},b={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?xd((e,t,n)=>{let r=P.runKernel(ni,y,b);return n([t,e,r]),m&&(r=G(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:v}})(p,f):xd((e,t,n,r)=>{let i=P.runKernel(ni,y,b);return r([t,e,i,n]),m&&(i=G(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:v}})(p,f,g)}const eg=/* @__PURE__ */ L({fusedDepthwiseConv2d_:$h});
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
function tg({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:i,activation:a=`linear`,preluActivationWeights:o,leakyreluAlpha:s=.2}){if(Kh(P.state.gradientDepth,a)===!1){let c=q(e,t,n,r);return i!=null&&(c=V(c,i)),Gh(c,a,o,s)}let c=I(e,`a`,`fused matMul`),l=I(t,`b`,`fused matMul`);[c,l]=N(c,l);let u=n?c.shape[c.rank-2]:c.shape[c.rank-1],d=r?l.shape[l.rank-1]:l.shape[l.rank-2],f=n?c.shape[c.rank-1]:c.shape[c.rank-2],p=r?l.shape[l.rank-2]:l.shape[l.rank-1],m=c.shape.slice(0,-2),h=l.shape.slice(0,-2),g=w(m),_=w(h);x(u===d,()=>`Error in fused matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${c.shape} and ${l.shape} and transposeA=${n} and transposeB=${r} must match.`);let v=Y(c.shape.slice(0,-2),l.shape.slice(0,-2)).concat([f,p]),y=n?G(c,[g,u,f]):G(c,[g,f,u]),b=r?G(l,[_,p,d]):G(l,[_,d,p]),S;i!=null&&(S=I(i,`bias`,`fused matMul`),[S]=N(S,c),Y(v,S.shape));let C;o!=null&&(C=I(o,`prelu weights`,`fused matMul`));let ee=(e,t)=>{let[o,s,c,l]=t,u=Uh(G(e,c.shape),c,a),d,f;if(!n&&!r?(d=q(u,s,!1,!0),f=q(o,u,!0,!1)):!n&&r?(d=q(u,s,!1,!1),f=q(u,o,!0,!1)):n&&!r?(d=q(s,u,!1,!0),f=q(o,u,!1,!1)):(d=q(s,u,!0,!0),f=q(u,o,!0,!0)),i!=null){let e=Wh(l,u);return[d,f,e]}return[d,f]},T={a:y,b,bias:S,preluActivationWeights:C},E={transposeA:n,transposeB:r,activation:a,leakyreluAlpha:s};return i==null?xd((e,t,n)=>{let r=P.runKernel(ei,T,E);return n([e,t,r]),{value:G(r,v),gradFunc:ee}})(y,b):xd((e,t,n,r)=>{let i=P.runKernel(ei,T,E);return r([e,t,i,n]),{value:G(i,v),gradFunc:ee}})(y,b,S)}const ng=/* @__PURE__ */ L({fusedMatMul_:tg});
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
var rg=/* @__PURE__ */ s({conv2d:()=>Jh,depthwiseConv2d:()=>eg,matMul:()=>ng});
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
function ig(e){return Rh(e,.54,.46)}const ag=/* @__PURE__ */ L({hammingWindow_:ig});
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
function og(e){return Rh(e,.5,.5)}const sg=/* @__PURE__ */ L({hannWindow_:og});
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
function cg(e,t,n,r=!1,i=0){let a=0,o=[];for(;a+t<=e.size;)o.push(J(e,a,t)),a+=n;if(r)for(;a<e.size;){let r=a+t-e.size,s=K([J(e,a,t-r),Zc([r],i)]);o.push(s),a+=n}return o.length===0?Ym([],[0,t]):G(K(o),[o.length,t])}const lg=/* @__PURE__ */ L({frame_:cg});
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
function ug(e,t,n,r,i=sg){r??=Lh(t);let a=lg(e,t,n),o=U(a,i(t));return Fm(o,r)}const dg=/* @__PURE__ */ L({stft_:ug});
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
function fg(e,t,n,r,i=`bilinear`,a=0){let o=I(e,`image`,`cropAndResize`),s=I(t,`boxes`,`cropAndResize`,`float32`),c=I(n,`boxInd`,`cropAndResize`,`int32`),l=s.shape[0];x(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),x(s.rank===2&&s.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${s.shape}.`),x(c.rank===1&&c.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${s.shape}.`),x(r.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),x(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),x(i===`bilinear`||i===`nearest`,()=>`method must be bilinear or nearest, but was ${i}`);let u={image:o,boxes:s,boxInd:c},d={method:i,extrapolationValue:a,cropSize:r};return P.runKernel(At,u,d)}const pg=/* @__PURE__ */ L({cropAndResize_:fg});
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
function mg(e){let t=I(e,`image`,`flipLeftRight`,`float32`);x(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);let n={image:t};return P.runKernel(Zt,n,{})}const hg=/* @__PURE__ */ L({flipLeftRight_:mg});
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
function gg(e){let t=I(e,`image`,`grayscaleToRGB`),n=t.rank-1,r=t.shape[n];x(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),x(r===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);let i=Array(t.rank);return i.fill(1,0,n),i[n]=3,Ru(t,i)}const _g=/* @__PURE__ */ L({grayscaleToRGB_:gg});
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
function vg(e){let t=I(e,`image`,`RGBToGrayscale`),n=t.rank-1,r=t.shape[n];x(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),x(r===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${r}.`);let i=t.dtype,a=B(t,`float32`),o=Jm([.2989,.587,.114]),s;switch(t.rank){case 2:s=eu(`ij,j->i`,a,o);break;case 3:s=eu(`ijk,k->ij`,a,o);break;case 4:s=eu(`ijkl,l->ijk`,a,o);break;case 5:s=eu(`ijklm,m->ijkl`,a,o);break;case 6:s=eu(`ijklmn,n->ijklm`,a,o);break;default:throw Error(`Not a valid tensor rank.`)}return s=Pu(s,-1),B(s,i)}const yg=/* @__PURE__ */ L({rgbToGrayscale_:vg});
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
function bg(e,t,n=0,r=.5){let i=I(e,`image`,`rotateWithOffset`,`float32`);x(i.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${i.rank}.`);let a={image:i},o={radians:t,fillValue:n,center:r};return P.runKernel($r,a,o)}const xg=/* @__PURE__ */ L({rotateWithOffset_:bg});
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
function Sg(e,t,n,r,i,a){r??=.5,i??=-1/0,a??=0;let o=e.shape[0];return n=Math.min(n,o),x(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),x(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),x(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),x(t.rank===1,()=>`scores must be a 1D tensor`),x(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),x(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a}}
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
function Cg(e,t,n,r=.5,i=-1/0){let a=I(e,`boxes`,`nonMaxSuppression`,`float32`),o=I(t,`scores`,`nonMaxSuppression`,`float32`),s=Sg(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c={maxOutputSize:n,iouThreshold:r,scoreThreshold:i};return P.runKernel(Bn,{boxes:a,scores:o},c)}const wg=/* @__PURE__ */ L({nonMaxSuppression_:Cg});
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
function Tg(e,t,n){let r=Eg(e,t,n),i=r<0?-(r+1):r;e.splice(i,0,t)}function Eg(e,t,n){return Og(e,t,n||Dg)}function Dg(e,t){return e>t?1:e<t?-1:0}function Og(e,t,n){let r=0,i=e.length,a=0,o=!1;for(;r<i;){a=r+(i-r>>>1);let s=n(t,e[a]);s>0?r=a+1:(i=a,o=!s)}return o?r:-r-1}
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
function kg(e,t,n,r,i){return Mg(e,t,n,r,i,0)}function Ag(e,t,n,r,i,a){return Mg(e,t,n,r,i,0,!1,a,!0)}function jg(e,t,n,r,i,a){return Mg(e,t,n,r,i,a,!0)}function Mg(e,t,n,r,i,a,o=!1,s=!1,c=!1){let l=[];for(let e=0;e<t.length;e++)t[e]>i&&l.push({score:t[e],boxIndex:e,suppressBeginIndex:0});l.sort(Fg);let u=a>0?-.5/a:0,d=[],f=[];for(;d.length<n&&l.length>0;){let t=l.pop(),{score:n,boxIndex:a,suppressBeginIndex:o}=t;if(n<i)break;let s=!1;for(let n=d.length-1;n>=o;--n){let o=Ng(e,a,d[n]);if(o>=r){s=!0;break}if(t.score*=Pg(r,u,o),t.score<=i)break}t.suppressBeginIndex=d.length,s||(t.score===n?(d.push(a),f.push(t.score)):t.score>i&&Tg(l,t,Fg))}let p=d.length,m=n-p;s&&m>0&&(d.push(...Array(m).fill(0)),f.push(...Array(m).fill(0)));let h={selectedIndices:d};return o&&(h.selectedScores=f),c&&(h.validOutputs=p),h}function Ng(e,t,n){let r=e.subarray(t*4,t*4+4),i=e.subarray(n*4,n*4+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),s=Math.max(r[0],r[2]),c=Math.max(r[1],r[3]),l=Math.min(i[0],i[2]),u=Math.min(i[1],i[3]),d=Math.max(i[0],i[2]),f=Math.max(i[1],i[3]),p=(s-a)*(c-o),m=(d-l)*(f-u);if(p<=0||m<=0)return 0;let h=Math.max(a,l),g=Math.max(o,u),_=Math.min(s,d),v=Math.min(c,f),y=Math.max(_-h,0)*Math.max(v-g,0);return y/(p+m-y)}function Pg(e,t,n){let r=Math.exp(t*n*n);return n<=e?r:0}function Fg(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}
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
async function Ig(e,t,n,r=.5,i=-1/0){let a=I(e,`boxes`,`nonMaxSuppressionAsync`),o=I(t,`scores`,`nonMaxSuppressionAsync`),s=Sg(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c=await Promise.all([a.data(),o.data()]),l=c[0],u=c[1],{selectedIndices:d}=kg(l,u,n,r,i);return a!==e&&a.dispose(),o!==t&&o.dispose(),Jm(d,`int32`)}const Lg=Ig;
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
function Rg(e,t,n,r=.5,i=-1/0,a=0){let o=I(e,`boxes`,`nonMaxSuppression`),s=I(t,`scores`,`nonMaxSuppression`),c=Sg(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l={boxes:o,scores:s},u={maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a},d=P.runKernel(Hn,l,u);return{selectedIndices:d[0],selectedScores:d[1]}}const zg=/* @__PURE__ */ L({nonMaxSuppressionWithScore_:Rg});
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
async function Bg(e,t,n,r=.5,i=-1/0,a=0){let o=I(e,`boxes`,`nonMaxSuppressionAsync`),s=I(t,`scores`,`nonMaxSuppressionAsync`),c=Sg(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l=await Promise.all([o.data(),s.data()]),u=l[0],d=l[1],{selectedIndices:f,selectedScores:p}=jg(u,d,n,r,i,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Jm(f,`int32`),selectedScores:Jm(p)}}const Vg=Bg;
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
function Hg(e,t,n,r=.5,i=-1/0,a=!1){let o=I(e,`boxes`,`nonMaxSuppression`),s=I(t,`scores`,`nonMaxSuppression`),c=Sg(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,f={boxes:o,scores:s},p={maxOutputSize:l,iouThreshold:u,scoreThreshold:d,padToMaxOutputSize:a},m=P.runKernel(Vn,f,p);return{selectedIndices:m[0],validOutputs:m[1]}}const Ug=/* @__PURE__ */ L({nonMaxSuppressionPadded_:Hg});
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
async function Wg(e,t,n,r=.5,i=-1/0,a=!1){let o=I(e,`boxes`,`nonMaxSuppressionAsync`),s=I(t,`scores`,`nonMaxSuppressionAsync`),c=Sg(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,[f,p]=await Promise.all([o.data(),s.data()]),{selectedIndices:m,validOutputs:h}=Ag(f,p,l,u,d,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Jm(m,`int32`),validOutputs:X(h,`int32`)}}const Gg=Wg;
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
function Kg(e,t,n=!1,r=!1){let i=I(e,`images`,`resizeBilinear`);x(i.rank===3||i.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${i.rank}.`),x(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),x(r===!1||n===!1,()=>`Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=G(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=P.runKernel(or,s,c);return o?G(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const qg=/* @__PURE__ */ L({resizeBilinear_:Kg});
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
function Jg(e,t,n=!1,r=!1){let i=I(e,`images`,`resizeNearestNeighbor`);x(i.rank===3||i.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${i.rank}.`),x(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),x(i.dtype===`float32`||i.dtype===`int32`,()=>"`images` must have `int32` or `float32` as dtype"),x(r===!1||n===!1,()=>`Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=G(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=P.runKernel(ir,s,c);return o?G(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const Yg=/* @__PURE__ */ L({resizeNearestNeighbor_:Jg});
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Xg(e,t=`binary`,n=!1,r=.5){let i=I(e,`image`,`threshold`),a=i.shape[0]*i.shape[1],o=U(Jm([r]),255),s,c,l,u;if(x(i.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${i.rank}.`),x(i.shape[2]===3||i.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${i.shape[2]}.`),x(i.dtype===`int32`||i.dtype===`float32`,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${i.dtype}.`),x(t===`otsu`||t===`binary`,()=>`Method must be binary or otsu, but was ${t}`),i.shape[2]===3){[s,c,l]=Nm(i,[1,1,1],-1);let e=U(s,.2989),t=U(c,.587),n=U(l,.114);u=V(V(e,t),n)}else u=e;t===`otsu`&&(o=Zg(Hc(B(tm(u),`int32`),Va([]),256),a));let d=n?cd(u,o):Ku(u,o);return B(U(d,255),`int32`)}function Zg(e,t){let n=Jm([-1]),r=Jm([0]),i=Jm([0]),a,o,s,c,l,u;for(let d=0;d<e.size-1;d++){a=J(e,0,d+1),o=J(e,d+1),l=H(Z(a),t),u=H(Z(o),t);let f=Z(U(a,Fp(0,a.size)));s=H(f,Z(a));let p=Zc(o.shape,a.size),m=V(Fp(0,o.size),p),h=U(o,m);c=H(Z(h),Z(o));let g=Q(s,c),_=Q(s,c),v=U(l,u);i=U(U(v,g),_);let y=Ku(i,r);r=Kl(y,i,r),n=Kl(y,Jm([d]),n)}return n}const Qg=/* @__PURE__ */ L({threshold_:Xg});
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
function $g(e,t,n=`nearest`,r=`constant`,i=0,a){let o=I(e,`image`,`transform`,`float32`),s=I(t,`transforms`,`transform`,`float32`);x(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),x(s.rank===2&&(s.shape[0]===o.shape[0]||s.shape[0]===1)&&s.shape[1]===8,()=>`Error in transform: Input transform should be batch x 8 or 1 x 8`),x(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);let c={image:o,transforms:s},l={interpolation:n,fillMode:r,fillValue:i,outputShape:a};return P.runKernel(Gr,c,l)}const e_=/* @__PURE__ */ L({transform_:$g});
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
function t_(e,t,n){let r=I(e,`a`,`bandPart`);x(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);let i=r.shape,[a,o]=r.shape.slice(-2),s,c;typeof t==`number`?(x(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),x(t<=a,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`),s=I(t<0?a:t,`numLower`,`bandPart`)):(x(t.dtype===`int32`,()=>`bandPart(): numLower's dtype must be an int32.`),s=Kl(od(t,0),a,sf(t,a))),typeof n==`number`?(x(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),x(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),c=I(n<0?o:n,`numUpper`,`bandPart`)):(x(n.dtype===`int32`,()=>`bandPart(): numUpper's dtype must be an int32.`),c=Kl(od(n,0),o,sf(n,o)));let l=G(Fp(0,a,1,`int32`),[-1,1]),u=Fp(0,o,1,`int32`),d=Q(l,u),f=Fd(cd(d,s),Ju(d,wd(c))),p=nf([a,o],r.dtype);return G(Vm(hh(G(r,[-1,a,o])).map(e=>Kl(f,e,p))),i)}const n_=/* @__PURE__ */ L({bandPart_:t_});
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
function r_(e){let t;if(Array.isArray(e)){t=!1,x(e!=null&&e.length>0,()=>`Gram-Schmidt process: input must not be null, undefined, or empty`);let n=e[0].shape[0];for(let t=1;t<e.length;++t)x(e[t].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[t].shape[0]} vs. ${n})`)}else t=!0,e=Nm(e,e.shape[0],0).map(e=>zm(e,[0]));x(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);let n=[],r=e;for(let t=0;t<e.length;++t)n.push(P.tidy(()=>{let e=r[t];if(t>0)for(let r=0;r<t;++r){let t=U(Z(U(n[r],e)),n[r]);e=Q(e,t)}return H(e,Ou(e,`euclidean`))}));return t?Vm(n,0):n}const i_=/* @__PURE__ */ L({gramSchmidt_:r_});
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
function a_(e,t=!1){if(x(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return o_(e,t);{let n=e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),r=hh(G(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),i=[],a=[];return r.forEach(e=>{let[n,r]=o_(e,t);i.push(n),a.push(r)}),[G(Vm(i,0),e.shape),G(Vm(a,0),e.shape)]}}function o_(e,t=!1){return P.tidy(()=>{x(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);let n=e.shape[0],r=e.shape[1],i=Bu(n),a=ys(e),o=Ym([[1]],[1,1]),s=ys(o),c=n>=r?r:n;for(let e=0;e<c;++e){let t=a,c=s,l=i;[s,a,i]=P.tidy(()=>{let t=J(a,[e,e],[n-e,1]),c=Ou(t),l=J(a,[e,e],[1,1]),u=Kl(Ku(l,0),Ym([[-1]]),Ym([[1]])),d=Q(l,U(u,c)),f=H(t,d);s=f.shape[0]===1?ys(o):K([o,J(f,[1,0],[f.shape[0]-1,f.shape[1]])],0);let p=wd(H(q(u,d),c)),m=J(a,[e,0],[n-e,r]),h=U(p,s),g=wh(s);if(e===0)a=Q(m,q(h,q(g,m)));else{let t=Q(m,q(h,q(g,m)));a=K([J(a,[0,0],[e,r]),t],0)}let _=wh(h),v=J(i,[0,e],[n,i.shape[1]-e]);if(e===0)i=Q(v,q(q(v,s),_));else{let t=Q(v,q(q(v,s),_));i=K([J(i,[0,0],[n,e]),t],1)}return[s,a,i]}),z([t,c,l])}return!t&&n>r&&(i=J(i,[0,0],[n,r]),a=J(a,[0,0],[r,r])),[i,a]})}const s_=/* @__PURE__ */ L({qr_:a_});
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
var $;(function(e){e[e.NONE=0]=`NONE`,e[e.MEAN=1]=`MEAN`,e[e.SUM=2]=`SUM`,e[e.SUM_BY_NONZERO_WEIGHTS=3]=`SUM_BY_NONZERO_WEIGHTS`})($||={});function c_(e,t,n=$.SUM_BY_NONZERO_WEIGHTS){let r=I(e,`losses`,`computeWeightedLoss`),i=null;t!=null&&(i=I(t,`weights`,`computeWeightedLoss`));let a=i==null?r:U(r,i);if(n===$.NONE)return a;if(n===$.SUM)return Z(a);if(n===$.MEAN){if(i==null)return tf(a);{let e=r.size/i.size,t=H(Z(a),Z(i));return e>1?H(t,X(e)):t}}if(n===$.SUM_BY_NONZERO_WEIGHTS){if(i==null)return H(Z(a),X(r.size));{let e=U(i,rf(r.shape)),t=B(Z(yf(e,X(0))),`float32`);return H(Z(a),t)}}throw Error(`Unknown reduction: ${n}`)}const l_=/* @__PURE__ */ L({computeWeightedLoss_:c_});
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
function u_(e,t,n,r=$.SUM_BY_NONZERO_WEIGHTS){let i=I(e,`labels`,`absoluteDifference`),a=I(t,`predictions`,`absoluteDifference`),o=null;n!=null&&(o=I(n,`weights`,`absoluteDifference`)),S(i.shape,a.shape,`Error in absoluteDifference: `);let s=W(Q(i,a));return l_(s,o,r)}const d_=/* @__PURE__ */ L({absoluteDifference_:u_});function f_(e,t,n,r,i=$.SUM_BY_NONZERO_WEIGHTS){let a=I(e,`labels`,`cosineDistance`),o=I(t,`predictions`,`cosineDistance`),s=null;r!=null&&(s=I(r,`weights`,`cosineDistance`)),S(a.shape,o.shape,`Error in cosineDistance: `);let c=X(1),l=Q(c,Z(U(a,o),n,!0));return l_(l,s,i)}const p_=/* @__PURE__ */ L({cosineDistance_:f_});function m_(e,t,n,r=$.SUM_BY_NONZERO_WEIGHTS){let i=I(e,`labels`,`hingeLoss`),a=I(t,`predictions`,`hingeLoss`),o=null;n!=null&&(o=I(n,`weights`,`hingeLoss`)),S(i.shape,a.shape,`Error in hingeLoss: `);let s=X(1);i=Q(U(X(2),i),s);let c=Vp(Q(s,U(i,a)));return l_(c,o,r)}const h_=/* @__PURE__ */ L({hingeLoss_:m_});
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
function g_(e,t,n,r=1,i=$.SUM_BY_NONZERO_WEIGHTS){let a=I(e,`labels`,`huberLoss`),o=I(t,`predictions`,`huberLoss`),s=null;n!=null&&(s=I(n,`weights`,`huberLoss`)),S(a.shape,o.shape,`Error in huberLoss: `);let c=X(r),l=W(Q(o,a)),u=sf(l,c),d=Q(l,u),f=V(U(X(.5),wu(u)),U(c,d));return l_(f,s,i)}const __=/* @__PURE__ */ L({huberLoss_:g_});
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
function v_(e,t,n,r=1e-7,i=$.SUM_BY_NONZERO_WEIGHTS){let a=I(e,`labels`,`logLoss`),o=I(t,`predictions`,`logLoss`),s=null;n!=null&&(s=I(n,`weights`,`logLoss`)),S(a.shape,o.shape,`Error in logLoss: `);let c=X(1),l=X(r),u=wd(U(a,pd(V(o,l)))),d=U(Q(c,a),pd(V(Q(c,o),l))),f=Q(u,d);return l_(f,s,i)}const y_=/* @__PURE__ */ L({logLoss_:v_});
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
function b_(e,t,n,r=$.SUM_BY_NONZERO_WEIGHTS){let i=I(e,`labels`,`meanSquaredError`),a=I(t,`predictions`,`meanSquaredError`),o=null;n!=null&&(o=I(n,`weights`,`meanSquaredError`)),S(i.shape,a.shape,`Error in meanSquaredError: `);let s=Lm(i,a);return l_(s,o,r)}const x_=/* @__PURE__ */ L({meanSquaredError_:b_});
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
function S_(e,t){let n=I(e,`labels`,`sigmoidCrossEntropyWithLogits`),r=I(t,`logits`,`sigmoidCrossEntropyWithLogits`);S(n.shape,r.shape,`Error in sigmoidCrossEntropyWithLogits: `);let i=Vp(r),a=U(r,n),o=hd(Mu(wd(W(r))));return V(Q(i,a),o)}function C_(e,t,n,r=0,i=$.SUM_BY_NONZERO_WEIGHTS){let a=I(e,`multiClassLabels`,`sigmoidCrossEntropy`),o=I(t,`logits`,`sigmoidCrossEntropy`),s=null;if(n!=null&&(s=I(n,`weights`,`sigmoidCrossEntropy`)),S(a.shape,o.shape,`Error in sigmoidCrossEntropy: `),r>0){let e=X(r),t=X(1),n=X(.5);a=V(U(a,Q(t,e)),U(n,e))}let c=S_(a,o);return l_(c,s,i)}const w_=/* @__PURE__ */ L({sigmoidCrossEntropy_:C_});
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
function T_(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return xd((e,t,r)=>{let i=Nd(t,[n],!0),a=Q(B(t,`float32`),i);r([e,a]);let o=wd(U(a,e));return{value:Z(o,[n]),gradFunc:(e,t)=>{let[r,i]=t,a=uu(e.shape,[n]);return[U(G(e,a),Q(B(r,`float32`),Mu(i))),U(G(e,a),Q(Mu(i),B(r,`float32`)))]}}})(e,t)}function E_(e,t,n,r=0,i=$.SUM_BY_NONZERO_WEIGHTS){let a=I(e,`onehotLabels`,`softmaxCrossEntropy`),o=I(t,`logits`,`softmaxCrossEntropy`),s=null;if(n!=null&&(s=I(n,`weights`,`softmaxCrossEntropy`)),S(a.shape,o.shape,`Error in softmaxCrossEntropy: `),r>0){let e=X(r),t=X(1),n=X(a.shape[1]);a=V(U(a,Q(t,e)),H(e,n))}let c=T_(a,o);return l_(c,s,i)}const D_=/* @__PURE__ */ L({softmaxCrossEntropy_:E_});
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
function O_(e,t,n,r){let i=I(e,`indices`,`sparseFillEmptyRows`,`int32`),a=I(t,`values`,`sparseFillEmptyRows`),o=I(n,`denseShape`,`sparseFillEmptyRows`,`int32`),s=I(r,`defaultValue`,`sparseFillEmptyRows`,a.dtype);if(i.rank!==2)throw Error(`Indices should be Tensor2D but received shape
        ${i.shape}`);if(a.rank!==1)throw Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(s.rank!==0)throw Error(`Default value should be a scalar but received shape ${s.shape}`);let c={indices:i,values:a,denseShape:o,defaultValue:s},l=P.runKernel(Or,c);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}const k_=/* @__PURE__ */ L({sparseFillEmptyRows_:O_});
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
function A_(e,t,n){let r=I(e,`inputIndices`,`sparseReshape`,`int32`),i=I(t,`inputShape`,`sparseReshape`,`int32`),a=I(n,`newShape`,`sparseReshape`,`int32`);if(r.rank!==2)throw Error(`Input indices should be Tensor2D but received shape
        ${r.shape}`);if(i.rank!==1)throw Error(`Input shape should be Tensor1D but received shape ${i.shape}`);if(a.rank!==1)throw Error(`New shape should be Tensor1D but received shape ${a.shape}`);let o={inputIndices:r,inputShape:i,newShape:a},s=P.runKernel(kr,o);return{outputIndices:s[0],outputShape:s[1]}}const j_=/* @__PURE__ */ L({sparseReshape_:A_});
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
function M_(e,t,n){let r=I(e,`data`,`sparseSegmentMean`),i=I(t,`indices`,`sparseSegmentMean`,`int32`),a=I(n,`segmentIds`,`sparseSegmentMean`,`int32`);if(r.rank<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.rank!==1)throw Error(`Indices should be Tensor1D but received shape
          ${i.shape}`);if(a.rank!==1)throw Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);let o={data:r,indices:i,segmentIds:a};return P.runKernel(Ar,o)}const N_=/* @__PURE__ */ L({sparseSegmentMean_:M_});
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
function P_(e,t,n){let r=I(e,`data`,`sparseSegmentSum`),i=I(t,`indices`,`sparseSegmentSum`,`int32`),a=I(n,`segmentIds`,`sparseSegmentSum`,`int32`);if(r.rank<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.rank!==1)throw Error(`Indices should be Tensor1D but received shape
         ${i.shape}`);if(a.rank!==1)throw Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);let o={data:r,indices:i,segmentIds:a};return P.runKernel(jr,o)}const F_=/* @__PURE__ */ L({sparseSegmentSum_:P_});
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
function I_(e,t,n,r,i,a,o,s){let c=I(e,`data`,`stringNGrams`,`string`);if(c.dtype!==`string`)throw Error(`Data must be of datatype string`);if(c.shape.length!==1)throw Error(`Data must be a vector, saw: ${c.shape}`);let l=I(t,`dataSplits`,`stringNGrams`);if(l.dtype!==`int32`)throw Error(`Data splits must be of datatype int32`);let u={separator:n,nGramWidths:r,leftPad:i,rightPad:a,padWidth:o,preserveShortSequences:s},d={data:c,dataSplits:l},f=P.runKernel(Lr,d,u);return{nGrams:f[0],nGramsSplits:f[1]}}const L_=/* @__PURE__ */ L({stringNGrams_:I_});
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
function R_(e,t,n=!0){let r=I(e,`input`,`stringSplit`,`string`),i=I(t,`delimiter`,`stringSplit`,`string`);if(r.rank!==1)throw Error(`Input should be Tensor1D but received shape ${r.shape}`);if(i.rank!==0)throw Error(`Delimiter should be a scalar but received shape ${i.shape}`);let a={skipEmpty:n},o={input:r,delimiter:i},s=P.runKernel(Rr,o,a);return{indices:s[0],values:s[1],shape:s[2]}}const z_=/* @__PURE__ */ L({stringSplit_:R_});
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
function B_(e,t){let n=I(e,`input`,`stringToHashBucketFast`,`string`),r={numBuckets:t};if(t<=0)throw Error(`Number of buckets must be at least 1`);let i={input:n};return P.runKernel(zr,i,r)}const V_=/* @__PURE__ */ L({stringToHashBucketFast_:B_});
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
function H_(e,t,n,r=!0){let i=I(e,`input`,`staticRegexReplace`,`string`),a={pattern:t,rewrite:n,replaceGlobal:r};return P.runKernel(Fr,{x:i},a)}const U_=/* @__PURE__ */ L({staticRegexReplace_:H_}),W_={fft:Dm,ifft:km,rfft:Fm,irfft:jm},G_={hammingWindow:ag,hannWindow:sg,frame:lg,stft:dg},K_={flipLeftRight:hg,grayscaleToRGB:_g,resizeNearestNeighbor:Yg,resizeBilinear:qg,rgbToGrayscale:yg,rotateWithOffset:xg,cropAndResize:pg,nonMaxSuppression:wg,nonMaxSuppressionAsync:Lg,nonMaxSuppressionWithScore:zg,nonMaxSuppressionWithScoreAsync:Vg,nonMaxSuppressionPadded:Ug,nonMaxSuppressionPaddedAsync:Gg,threshold:Qg,transform:e_},q_={bandPart:n_,gramSchmidt:i_,qr:s_},J_={absoluteDifference:d_,computeWeightedLoss:l_,cosineDistance:p_,hingeLoss:h_,huberLoss:__,logLoss:y_,meanSquaredError:x_,sigmoidCrossEntropy:w_,softmaxCrossEntropy:D_},Y_={sparseFillEmptyRows:k_,sparseReshape:j_,sparseSegmentMean:N_,sparseSegmentSum:F_},X_={stringNGrams:L_,stringSplit:z_,stringToHashBucketFast:V_,staticRegexReplace:U_}
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
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Z_=/* @__PURE__ */ s({Serializable:()=>ev,SerializationMap:()=>tv,getRegisteredName:()=>rv,registerClass:()=>nv});const Q_=/* @__PURE__ */ new Map,$_=/* @__PURE__ */ new Map;var ev=class{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}},tv=class e{constructor(){this.classNameMap={}}static getMap(){return e.instance??=new e,e.instance}static register(t){e.getMap().classNameMap[t.className]=[t,t.fromConfig]}};function nv(e,t,n){x(e.className!=null,()=>`Class being registered does not have the static className property defined.`),x(typeof e.className==`string`,()=>`className is required to be a string, but got type `+typeof e.className),x(e.className.length>0,()=>`Class being registered has an empty-string as its className, which is disallowed.`),t===void 0&&(t=`Custom`),n===void 0&&(n=e.className);let r=n,i=t+`>`+r;return tv.register(e),Q_.set(i,e),$_.set(e,i),e}function rv(e){return $_.has(e)?$_.get(e):e.className}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var iv=class extends ev{minimize(e,t=!1,n){let{value:r,grads:i}=this.computeGradients(e,n);if(n!=null){let e=n.map(e=>({name:e.name,tensor:i[e.name]}));this.applyGradients(e)}else this.applyGradients(i);return z(i),t?r:(r.dispose(),null)}get iterations(){return this.iterations_??=0,this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return bd(e,t)}dispose(){this.iterations_!=null&&z(this.iterations_)}async saveIterations(){return this.iterations_??=0,{name:`iter`,tensor:X(this.iterations_,`int32`)}}async getWeights(){throw Error(`getWeights() is not implemented for this optimizer yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}};Object.defineProperty(iv,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var av=class extends iv{static get className(){return`Adadelta`}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],n??(this.epsilon=P.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=P.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:R(()=>Jl(r).variable(!1))}),this.accumulatedUpdates[n]??(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:R(()=>Jl(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable,o=this.accumulatedUpdates[n].variable;R(()=>{let e=V(U(a,this.rho),U(wu(i),1-this.rho)),t=U(H(Su(V(o,this.epsilon)),Su(V(a,this.epsilon))),i),n=V(U(o,this.rho),U(wu(t),1-this.rho));a.assign(e),o.assign(n);let s=V(U(t,-this.learningRate),r);r.assign(s)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(z(this.accumulatedGrads.map(e=>e.variable)),z(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){let e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=e.length/2;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedUpdates=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}},ov=class extends iv{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
return`Adagrad`}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=P.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:R(()=>Zc(r.shape,this.initialAccumulatorValue).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable;R(()=>{let e=V(a,wu(i));a.assign(e);let t=V(U(H(i,Su(V(e,P.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&z(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}},sv=class extends iv{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
return`Adam`}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],R(()=>{this.accBeta1=X(t).variable(),this.accBeta2=X(n).variable()}),r??(this.epsilon=P.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);R(()=>{let n=Q(1,this.accBeta1),r=Q(1,this.accBeta2);t.forEach((t,i)=>{let a=P.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:R(()=>Jl(a).variable(!1))}),this.accumulatedSecondMoment[i]??(this.accumulatedSecondMoment[i]={originalName:`${t}/v`,variable:R(()=>Jl(a).variable(!1))});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedSecondMoment[i].variable,l=V(U(s,this.beta1),U(o,1-this.beta1)),u=V(U(c,this.beta2),U(wu(o),1-this.beta2)),d=H(l,n),f=H(u,r);s.assign(l),c.assign(u);let p=V(U(H(d,V(Su(f),this.epsilon)),-this.learningRate),a);a.assign(p)}),this.accBeta1.assign(U(this.accBeta1,this.beta1)),this.accBeta2.assign(U(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&z(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&z(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),R(()=>{this.accBeta1.assign(bu(this.beta1,this.iterations_+1)),this.accBeta2.assign(bu(this.beta2,this.iterations_+1))});let t=e.length/2;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}},cv=class extends iv{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
return`Adamax`}constructor(e,t,n,r=null,i=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=i,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],R(()=>{this.iteration=X(0).variable(),this.accBeta1=X(t).variable()}),r??(this.epsilon=P.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);R(()=>{let n=Q(1,this.accBeta1),r=H(-this.learningRate,V(U(this.iteration,this.decay),1));t.forEach((t,i)=>{let a=P.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:Jl(a).variable(!1)}),this.accumulatedWeightedInfNorm[i]??(this.accumulatedWeightedInfNorm[i]={originalName:`${t}/v`,variable:Jl(a).variable(!1)});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedWeightedInfNorm[i].variable,l=V(U(s,this.beta1),U(o,1-this.beta1)),u=U(c,this.beta2),d=W(o),f=$d(u,d);s.assign(l),c.assign(f);let p=V(U(H(r,n),H(l,V(f,this.epsilon))),a);a.assign(p)}),this.iteration.assign(V(this.iteration,1)),this.accBeta1.assign(U(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&z(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&z(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw Error(`getWeights() is not implemented for Adamax yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for Adamax yet.`)}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}},lv=class extends iv{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
return`SGD`}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=Array.isArray(e)?e[n].tensor:e[t];if(r==null)return;let i=P.registeredVariables[t];R(()=>{let e=V(U(this.c,r),i);i.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=$a(X(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw Error(`SGD optimizer does not have settable weights.`)}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}},uv=class extends lv{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
return`Momentum`}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=X(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=P.registeredVariables[t];this.accumulations[n]??(this.accumulations[n]={originalName:`${t}/momentum`,variable:R(()=>Jl(r).variable(!1))});let i=this.accumulations[n].variable,a=Array.isArray(e)?e[n].tensor:e[t];a!=null&&R(()=>{let e,t=V(U(this.m,i),a);e=this.useNesterov?V(U(this.c,V(a,U(t,this.m))),r):V(U(this.c,t),r),i.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&z(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}},dv=class extends iv{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
return`RMSProp`}constructor(e,t=.9,n=0,r=null,i=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=i,r??(this.epsilon=P.backend.epsilon()),e==null)throw Error(`learningRate for RMSPropOptimizer must be defined.`)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=P.registeredVariables[t];this.accumulatedMeanSquares[n]??(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:R(()=>Jl(r).variable(!1))}),this.accumulatedMoments[n]??(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:R(()=>Jl(r).variable(!1))}),this.accumulatedMeanGrads[n]==null&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:R(()=>Jl(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedMeanSquares[n].variable,o=this.accumulatedMoments[n].variable;R(()=>{let e=V(U(a,this.decay),U(wu(i),1-this.decay));if(this.centered){let t=this.accumulatedMeanGrads[n].variable,s=V(U(t,this.decay),U(i,1-this.decay)),c=H(U(i,this.learningRate),Su(Q(e,V(wu(s),this.epsilon)))),l=V(U(o,this.momentum),c);a.assign(e),t.assign(s),o.assign(l);let u=Q(r,l);r.assign(u)}else{let e=V(U(a,this.decay),U(wu(i),1-this.decay)),t=V(U(o,this.momentum),H(U(i,this.learningRate),Su(V(e,this.epsilon))));a.assign(e),o.assign(t);let n=Q(r,t);r.assign(n)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&z(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&z(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&z(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=this.centered?e.length/3:e.length/2;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedMoments=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}};
/**
* @license
* Copyright 2022 Google LLC.
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
const fv=[av,ov,sv,cv,uv,dv,lv];function pv(){for(let e of fv)nv(e)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function mv(e){return new Promise(e=>setTimeout(e)).then(e)}var hv=class e{constructor(t){if(!D().getBool(`IS_BROWSER`))throw Error(`browserDownloads() cannot proceed because the current environment is not a browser.`);t.startsWith(e.URL_SCHEME)&&(t=t.slice(e.URL_SCHEME.length)),(t==null||t.length===0)&&(t=`model`),this.modelJsonFileName=t+`.json`,this.weightDataFileName=t+`.weights.bin`}async save(e){if(typeof document>`u`)throw Error("Browser downloads are not supported in this environment since `document` is not present");let t=Ua.join(e.weightData),n=window.URL.createObjectURL(new Blob([t],{type:`application/octet-stream`}));if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserDownloads.save() does not support saving model topology in binary formats yet.`);{let t=To(e,[{paths:[`./`+this.weightDataFileName],weights:e.weightSpecs}]),r=window.URL.createObjectURL(new Blob([JSON.stringify(t)],{type:`application/json`})),i=this.modelJsonAnchor==null?document.createElement(`a`):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=r,await mv(()=>i.dispatchEvent(new MouseEvent(`click`))),e.weightData!=null){let e=this.weightDataAnchor==null?document.createElement(`a`):this.weightDataAnchor;e.download=this.weightDataFileName,e.href=n,await mv(()=>e.dispatchEvent(new MouseEvent(`click`)))}return{modelArtifactsInfo:Oo(e)}}}};hv.URL_SCHEME=`downloads://`;var gv=class{constructor(e){if(e==null||e.length<1)throw Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{let n=new FileReader;n.onload=n=>{let r=JSON.parse(n.target.result),i=r.modelTopology;if(i==null){t(/* @__PURE__ */ Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(r.weightsManifest==null){t(/* @__PURE__ */ Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:i});return}e(Do(r,e=>this.loadWeights(e)))},n.onerror=e=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),n.readAsText(this.jsonFile)})}loadWeights(e){let t=[],n=[];for(let r of e)t.push(...r.weights),n.push(...r.paths);let r=this.checkManifestAndWeightFiles(e),i=n.map(e=>this.loadWeightsFile(e,r[e]));return Promise.all(i).then(e=>[t,e])}loadWeightsFile(e,t){return new Promise((n,r)=>{let i=new FileReader;i.onload=e=>{let t=e.target.result;n(t)},i.onerror=t=>r(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){let t=[],n=this.weightsFiles.map(e=>wo(e.name)),r={};for(let i of e)i.paths.forEach(e=>{let i=wo(e);if(t.indexOf(i)!==-1)throw Error(`Duplicate file basename found in weights manifest: '${i}'`);if(t.push(i),n.indexOf(i)===-1)throw Error(`Weight file with basename '${i}' is not provided.`);r[e]=this.weightsFiles[n.indexOf(i)]});if(t.length!==this.weightsFiles.length)throw Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}};Po.registerSaveRouter(e=>D().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(hv.URL_SCHEME)?_v(e.slice(hv.URL_SCHEME.length)):null);function _v(e=`model`){return new hv(e)}function vv(e){return new gv(e)}
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
function yv(e,t,n,r){o(e),n??=0,r??=1,s(n,r);let i=0,a=a=>(a.then(a=>(t(n+ ++i/e.length*(r-n)),a)),a);function o(e){x(e!=null&&Array.isArray(e)&&e.length>0,()=>`promises must be a none empty array`)}function s(e,t){x(e>=0&&e<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${e}`),x(t>=0&&t<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${t}`),x(t>=e,()=>`startFraction must be no more than endFraction, but got startFraction ${e} and endFraction ${t}`)}return Promise.all(e.map(a))}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
async function bv(e,t){t??={};let n=t.fetchFunc==null?D().platform.fetch:t.fetchFunc,r=e.map(e=>n(e,t.requestInit,{isBinary:!0})),i=(t.onProgress==null?await Promise.all(r):await yv(r,t.onProgress,0,.5)).map(e=>e.arrayBuffer());return t.onProgress==null?await Promise.all(i):await yv(i,t.onProgress,.5,1)}function xv(e,t){var n;let r=t.fetchFunc==null?D().platform.fetch:t.fetchFunc,i=0,a;return(n=t.onProgress)==null||n.call(t,0),new ReadableStream({pull:async n=>{for(var o;i<e.length;){a||=(await r(e[i],t.requestInit,{isBinary:!0})).body.getReader();let{done:s,value:c}=await a.read();if(s){i++,a=void 0,(o=t.onProgress)==null||o.call(t,i/e.length);continue}n.enqueue(c);return}n.close()}})}async function Sv(e,t=``,n,r){return Cv(e=>bv(e,{requestInit:r}))(e,t,n)}function Cv(e){return async(t,n=``,r)=>{let i=t.map(()=>!1),a={},o=r==null?[]:r.map(()=>!1),s=[];if(t.forEach((e,t)=>{let n=0;e.weights.forEach(e=>{let c=`quantization`in e?e.quantization.dtype:e.dtype,l=Ha[c]*w(e.shape),u=()=>{i[t]=!0,a[t]??(a[t]=[]),a[t].push({manifestEntry:e,groupOffset:n,sizeBytes:l})};r==null?u():r.forEach((t,n)=>{t===e.name&&(u(),o[n]=!0)}),s.push(e.name),n+=l})}),!o.every(e=>e)){let e=r.filter((e,t)=>!o[t]);throw Error(`Could not find weights in manifest with names: ${e.join(`, `)}. \nManifest JSON has weights with names: ${s.join(`, `)}.`)}let c=i.reduce((e,t,n)=>(t&&e.push(n),e),[]),l=[];c.forEach(e=>{t[e].paths.forEach(e=>{let t=n+(n.endsWith(`/`)?``:`/`)+e;l.push(t)})});let u=await e(l),d={},f=0;return c.forEach(e=>{let n=t[e].paths.length,r=new Ua(u.slice(f,f+n));a[e].forEach(e=>{let t=fo(r.slice(e.groupOffset,e.groupOffset+e.sizeBytes),[e.manifestEntry]);for(let e in t)d[e]=t[e]}),f+=n}),d}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var wv=class{constructor(e,t){if(this.DEFAULT_METHOD=`POST`,t??={},this.weightPathPrefix=t.weightPathPrefix,this.weightUrlConverter=t.weightUrlConverter,t.fetchFunc==null?this.fetch=D().platform.fetch:(x(typeof t.fetchFunc==`function`,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc),x(e!=null&&e.length>0,()=>`URL path for http must not be null, undefined or empty.`),Array.isArray(e)&&x(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,t.requestInit!=null&&t.requestInit.body!=null)throw Error(`requestInit is expected to have no pre-existing body, but has one.`);this.requestInit=t.requestInit||{},this.loadOptions=t}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.`);let t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;let n=To(e,[{paths:[`./model.weights.bin`],weights:e.weightSpecs}]);if(t.body.append(`model.json`,new Blob([JSON.stringify(n)],{type:`application/json`}),`model.json`),e.weightData!=null){let n=Ua.join(e.weightData);t.body.append(`model.weights.bin`,new Blob([n],{type:`application/octet-stream`}),`model.weights.bin`)}let r=await this.fetch(this.path,t);if(r.ok)return{modelArtifactsInfo:Oo(e),responses:[r]};throw Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async loadModelJSON(){let e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch{let e=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(`.pb`)?e+=` Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.`:e+=` Please make sure the server is serving valid JSON for this request.`,Error(e)}let n=t.modelTopology,r=t.weightsManifest;if(n==null&&r==null)throw Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return t}async load(){return this.loadOptions.streamWeights?this.loadStream():Do(await this.loadModelJSON(),e=>this.loadWeights(e))}async loadStream(){let e=await this.loadModelJSON(),t=await this.getWeightUrls(e.weightsManifest),n=ko(e.weightsManifest);return Object.assign(Object.assign({},e),{weightSpecs:n,getWeightStream:()=>xv(t,this.loadOptions)})}async getWeightUrls(e){let[t,n]=Tv(Array.isArray(this.path)?this.path[1]:this.path),r=this.weightPathPrefix||t,i=[],a=[];for(let t of e)for(let e of t.paths)this.weightUrlConverter==null?i.push(r+e+n):a.push(this.weightUrlConverter(e));return this.weightUrlConverter&&i.push(...await Promise.all(a)),i}async loadWeights(e){let t=await this.getWeightUrls(e);return[ko(e),await bv(t,this.loadOptions)]}};wv.URL_SCHEME_REGEX=/^https?:\/\//;function Tv(e){let t=e.lastIndexOf(`/`),n=e.lastIndexOf(`?`),r=e.substring(0,t),i=n>t?e.substring(n):``;return[r+`/`,i]}function Ev(e){return e.match(wv.URL_SCHEME_REGEX)!=null}const Dv=(e,t)=>{if(typeof fetch>`u`&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(n=Array.isArray(e)?e.every(e=>Ev(e)):Ev(e),n)return Ov(e,t)}return null};Po.registerSaveRouter(Dv),Po.registerLoadRouter(Dv);function Ov(e,t){return new wv(e,t)}function kv(e,t){return Ov(e,t)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Av=class{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}},jv=class{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}},Mv=class{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}};function Nv(e,t,n,r){return new Mv(Pv(...arguments))}function Pv(e,t,n,r){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new Av(e):(console.warn(`Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release.`),new Av({modelTopology:e})):(console.warn(`Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release.`),new Av({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r}))}function Fv(e){return new jv(e)}function Iv(e){return new jv(e)}var Lv=/* @__PURE__ */ s({CompositeArrayBuffer:()=>Ua,browserFiles:()=>vv,browserHTTPRequest:()=>kv,concatenateArrayBuffers:()=>Co,copyModel:()=>ls,decodeWeights:()=>fo,decodeWeightsStream:()=>_o,encodeWeights:()=>uo,fromMemory:()=>Nv,fromMemorySync:()=>Pv,getLoadHandlers:()=>Ro,getModelArtifactsForJSON:()=>Do,getModelArtifactsForJSONSync:()=>Eo,getModelArtifactsInfoForJSON:()=>Oo,getSaveHandlers:()=>Lo,getWeightSpecs:()=>ko,http:()=>Ov,isHTTPScheme:()=>Ev,listModels:()=>ss,loadWeights:()=>Sv,moveModel:()=>us,registerLoadRouter:()=>Io,registerSaveRouter:()=>Fo,removeModel:()=>cs,weightsLoaderFactory:()=>Cv,withSaveHandler:()=>Fv,withSaveHandlerSync:()=>Iv});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Rv(e,t,n){let r=I(e,`labels`,`confusionMatrix`),i=I(t,`predictions`,`confusionMatrix`);x(n==null||n>0&&Number.isInteger(n),()=>`If provided, numClasses must be a positive integer, but got ${n}`),x(r.rank===1,()=>`Expected the rank of labels to be 1, but got ${r.rank}`),x(i.rank===1,()=>`Expected the rank of predictions to be 1, but got ${i.rank}`),x(r.shape[0]===i.shape[0],()=>`Mismatch in the number of examples: ${r.shape[0]} vs. ${i.shape[0]}. Labels and predictions should have the same number of elements.`),x(n>0&&Number.isInteger(n),()=>`numClasses is required to be a positive integer, but got ${n}`);let a=xf(B(r,`int32`),n),o=xf(B(i,`int32`),n),s=wh(a),c=q(s,o);return B(c,`int32`)}const zv=/* @__PURE__ */ L({confusionMatrix_:Rv});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Bv=/* @__PURE__ */ s({confusionMatrix:()=>zv}),Vv=/* @__PURE__ */ s({draw:()=>$v,fromPixels:()=>ey,fromPixelsAsync:()=>Yv,toPixels:()=>Qv})
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
;let Hv,Uv=!1;function Wv(e,t=3){if(t>4)throw Error(`Cannot construct Tensor with more than 4 channels from pixels.`);if(e==null)throw Error(`pixels passed to tf.browser.fromPixels() can not be null`);let n=!1,r=!1,i=!1,a=!1,o=!1,s=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<`u`&&e instanceof ImageData)r=!0;else if(typeof HTMLVideoElement<`u`&&e instanceof HTMLVideoElement)i=!0;else if(typeof HTMLImageElement<`u`&&e instanceof HTMLImageElement)a=!0;else if(e.getContext!=null)o=!0;else if(typeof ImageBitmap<`u`&&e instanceof ImageBitmap)s=!0;else throw Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(si(`FromPixels`,P.backendName)!=null){let n={pixels:e},r={numChannels:t};return P.runKernel(Qr,n,r)}let[c,l]=i?[e.videoWidth,e.videoHeight]:[e.width,e.height],u;if(o)u=e.getContext(`2d`).getImageData(0,0,c,l).data;else if(r||n)u=e.data;else if(a||i||s){if(Hv==null){if(typeof document>`u`){if(typeof OffscreenCanvas<`u`&&typeof OffscreenCanvasRenderingContext2D<`u`)Hv=new OffscreenCanvas(1,1).getContext(`2d`);else throw Error(`Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.`)}else Hv=document.createElement(`canvas`).getContext(`2d`,{willReadFrequently:!0})}Hv.canvas.width=c,Hv.canvas.height=l,Hv.drawImage(e,0,0,c,l),u=Hv.getImageData(0,0,c,l).data}let d;if(t===4)d=new Int32Array(u);else{let e=c*l;d=new Int32Array(e*t);for(let n=0;n<e;n++)for(let e=0;e<t;++e)d[n*t+e]=u[n*4+e]}return Xm(d,[l,c,t],`int32`)}function Gv(e){return e!=null&&e.data instanceof Uint8Array}function Kv(){return typeof window<`u`&&typeof ImageBitmap<`u`&&window.hasOwnProperty(`createImageBitmap`)}function qv(e){return e!=null&&e.width!==0&&e.height!==0}function Jv(e){return Kv()&&!(e instanceof ImageBitmap)&&qv(e)&&!Gv(e)}async function Yv(e,t=3){let n=null;if(D().getBool(`WRAP_TO_IMAGEBITMAP`)&&Jv(e)){let t;try{t=await createImageBitmap(e,{premultiplyAlpha:`none`})}catch{t=null}n=t!=null&&t.width===e.width&&t.height===e.height?t:e}else n=e;return Wv(n,t)}function Xv(e){if(e.rank!==2&&e.rank!==3)throw Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);let t=e.rank===2?1:e.shape[2];if(t>4||t===2)throw Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if(e.dtype!==`float32`&&e.dtype!==`int32`)throw Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}function Zv(e){let t=e?.alpha||1;if(t>1||t<0)throw Error(`Alpha value ${t} is suppoed to be in range [0 - 1].`)}async function Qv(e,t){let n=I(e,`img`,`toPixels`);if(!(e instanceof M)){let e=n;n=B(e,`int32`),e.dispose()}Xv(n);let[r,i]=n.shape.slice(0,2),a=n.rank===2?1:n.shape[2],o=await n.data(),s=n.dtype===`float32`?255:1,c=new Uint8ClampedArray(i*r*4);for(let e=0;e<r*i;++e){let t=[0,0,0,255];for(let r=0;r<a;r++){let i=o[e*a+r];if(n.dtype===`float32`){if(i<0||i>1)throw Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${i}.`)}else if(n.dtype===`int32`&&(i<0||i>255))throw Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${i}.`);a===1?(t[0]=i*s,t[1]=i*s,t[2]=i*s):t[r]=i*s}let r=e*4;c[r+0]=Math.round(t[0]),c[r+1]=Math.round(t[1]),c[r+2]=Math.round(t[2]),c[r+3]=Math.round(t[3])}if(t!=null){Uv||si(`Draw`,P.backendName)!=null&&(console.warn(`tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead.`),Uv=!0),t.width=i,t.height=r;let e=t.getContext(`2d`),n=new ImageData(c,i,r);e.putImageData(n,0,0)}return n!==e&&n.dispose(),c}function $v(e,t,n){let r=I(e,`img`,`draw`);if(!(e instanceof M)){let e=r;r=B(e,`int32`),e.dispose()}Xv(r),Zv(n?.imageOptions);let i={image:r},a={canvas:t,options:n};P.runKernel(Bt,i,a)}const ey=/* @__PURE__ */ L({fromPixels_:Wv});var ty=/* @__PURE__ */ s({prepareAndValidate:()=>ny});function ny(e,t){let n=e.shape.length,r=t.shape.length;if(n<1)throw Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!==`int32`)throw Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(w(e.shape)===0)throw Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);let i=t.shape,a=i[i.length-1],o=1;for(let e=0;e<i.length-1;++e)o*=i[e];let s=e.shape,c=i.slice();c.pop();let l=1;for(let e=a;e<n;++e)l*=s[e],c.push(s[e]);let u=[...Ce(e.shape).map(e=>e/l),1].slice(0,a);return[c,o,l,u]}
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
var ry=/* @__PURE__ */ s({assertParamsValid:()=>iy,computeFlatOffset:()=>_y,computeOutShape:()=>oy,getNormalizedAxes:()=>uy,isSliceContinous:()=>gy,maskToAxes:()=>ay,parseSliceParams:()=>vy,sliceInfo:()=>yy,startForAxis:()=>my,startIndicesWithElidedDims:()=>dy,stopForAxis:()=>hy,stopIndicesWithElidedDims:()=>fy,stridesForAxis:()=>py,stridesWithElidedDims:()=>sy});function iy(e,t,n){let r=e.shape.length;x(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),x(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let i=0;i<r;++i)x(t[i]+n[i]<=e.shape[i],()=>`Error in slice${r}D: begin[${i}] + size[${i}] (${t[i]+n[i]}) would overflow input.shape[${i}] (${e.shape[i]})`)}function ay(e){let t=[],n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function oy(e,t,n){let r=[];for(let i=0;i<e.length;i++)r[i]=Math.ceil((t[i]-e[i])/n[i]);return r}function sy(e,t,n,r){let i=[...e];for(let e=i.length;e<r.length;e++)i.push(1);for(let e=0;e<n;e++)e===0?i[t]=1:(i.splice(t,0,1),i.pop());return i}function cy(e,t,n){return n<=e?n:n-(t-1)}function ly(e,t){let n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function uy(e,t,n,r,i,a,o,s,c){let l=e.length,u=Array(l),d=Array(l),f=Array(l);if(t.length&&n>0){let c=t[0],l=n+1;u=dy(o,c,l,r,e),d=fy(s,c,l,i,e),f=sy(a,c,l,e)}else for(let t=0;t<l;t++)u[t]=my(o,r,a,e,t,c),d[t]=hy(s,i,a,e,t,c),f[t]=py(a,t,c);return{begin:u,end:d,strides:f}}function dy(e,t,n,r,i){let a=[...i],o=ly(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{let o=cy(t,n,i),s=r[o];e&1<<o&&(s=0),a[i]=s}return a}function fy(e,t,n,r,i){let a=[...i],o=ly(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=2**53-1;else{let o=cy(t,n,i),s=r[o];e&1<<o&&(s=2**53-1),a[i]=s}for(let e=0;e<a.length;e++){let t=i[e];a[e]<0&&(a[e]+=t),a[e]=h(0,a[e],i[e])}return a}function py(e,t,n){let r=e[t];return(n&1<<t||r==null)&&(r=1),r}function my(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?-(2**53-1):2**53-1);let c=r[i];return o<0&&(o+=c),o=h(0,o,c-1),o}function hy(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?2**53-1:-(2**53-1));let c=r[i];return o<0&&(o+=c),o=s>0?h(0,o,c):h(-1,o,c-1),o}function gy(e,t,n){let r=n.length;for(let e=0;e<n.length;e++)if(n[e]>1){r=e;break}for(let i=r+1;i<n.length;i++)if(t[i]>0||n[i]!==e[i])return!1;return!0}function _y(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function vy(e,t,n){let r,i=e.shape.length;r=typeof t==`number`?[t,...Array(i-1).fill(0)]:t.length<i?t.concat(Array(i-t.length).fill(0)):t.slice(),r.forEach(e=>{x(e!==-1,()=>`slice() does not support negative begin indexing.`)});let a;return a=n==null?Array(i).fill(-1):typeof n==`number`?[n,...Array(i-1).fill(-1)]:n.length<i?n.concat(Array(i-n.length).fill(-1)):n,a=a.map((t,n)=>t>=0?t:(x(t===-1,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,a]}function yy(e,t,n,r,i,a,o,s,c){let l;if(r==null?(l=Array(t.length),l.fill(1)):l=r,o!=null&&o&o-1)throw Error(`Multiple ellipses in slice is not allowed.`);let u=!1,d={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};for(let e=0;e<d.dims;e++)u&&1<<e&s&&d.numAddAxisAfterEllipsis++,1<<e&o&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);let f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};by(d,f);let p=!0,m=!0,h=!0,g=[],_=[];for(let t=0;t<e.length;++t){if(f.strides[t]===0)throw Error(`strides[${t}] must be non-zero`);let n=!!(f.shrinkAxisMask&1<<t),r=e[t];if(r===-1){g.push(n?1:-1);continue}let i=[f.beginMask&1<<t,f.endMask&1<<t],a=[f.strides[t]>0?0:-1,f.strides[t]>0?r:r-1];if(n&&f.strides[t]<=0)throw Error(`only stride 1 allowed on non-range indexing.`);h&&=f.strides[t]===1;let o=!!(f.beginMask&1<<t&&f.endMask&1<<t);if(f.beginValid&&f.endValid){if(n){let e=f.begin[t]<0?r+f.begin[t]:f.begin[t];if(f.begin[t]=e,f.end[t]=f.begin[t]+1,e<0||e>=r)throw Error(`slice index ${f.begin[t]} of dimension ${t} out of bounds.`)}else f.begin[t]=xy(f.begin[t],0,f.strides[t],r,i,a),f.end[t]=xy(f.end[t],1,f.strides[t],r,i,a);let e=f.strides[t]===1&&f.begin[t]===0&&f.end[t]===r;p&&=e,m&&=t===0&&f.strides[t]===1||e}else p=p&&f.strides[t]===1&&o,m&&=t===0&&f.strides[t]===1||o;let s,c=!1;if(f.beginValid&&f.endValid?(s=f.end[t]-f.begin[t],c=!0):n?(s=1,c=!0):o&&r>=0&&(s=f.strides[t]<0?-r:r,c=!0),c){let e;e=s===0||s<0!=f.strides[t]<0?0:Math.trunc(s/f.strides[t])+(s%f.strides[t]===0?0:1),g.push(e)}else g.push(-1)}for(let e=0;e<f.finalShapeGatherIndices.length;++e){let t=f.finalShapeGatherIndices[e];t>=0?_.push(g[t]):t===-2&&_.push(1)}return{finalShapeSparse:_.filter((e,t)=>f.finalShapeGatherIndices[t]!==-2),finalShape:_,isIdentity:p,sliceDim0:m,isSimpleSlice:h,begin:f.begin,end:f.end,strides:f.strides}}function by(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=Array(t.dims),t.end=Array(t.dims),t.strides=Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){let i=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<i;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(-2),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(-1),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function xy(e,t,n,r,i,a){if(i[t])return n>0?a[t]:a[t+1&1];{let t=e<0?r+e:e;return t<a[0]?a[0]:t>a[1]?a[1]:t}}
/** @license See the LICENSE file. */
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var Sy=class{static sgd(e){return new lv(e)}static momentum(e,t,n=!1){return new uv(e,t,n)}static rmsprop(e,t=.9,n=0,r=null,i=!1){return new dv(e,t,n,r,i)}static adam(e=.001,t=.9,n=.999,r=null){return new sv(e,t,n,r)}static adadelta(e=.001,t=.95,n=null){return new av(e,t,n)}static adamax(e=.002,t=.9,n=.999,r=null,i=0){return new cv(e,t,n,r,i)}static adagrad(e,t=.1){return new ov(e,t)}};
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
const Cy=Sy,wy=typeof requestAnimationFrame<`u`?requestAnimationFrame:typeof setImmediate<`u`?setImmediate:e=>e()
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
;function Ty(){return new Promise(e=>wy(()=>e()))}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
function Ey(e,t){let n=e[0].length;e.forEach((e,t)=>{x(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),x(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);let r=e[0];e.forEach((e,i)=>{for(let a=0;a<n;a++)x(a===t||e[a]===r[a],()=>`Error in concat${n}D: Shape of tensors[${i}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${i}.`)})}function Dy(e,t){let n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}
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
var Oy;(function(e){e[e.FIRST_DIM_SIZE=0]=`FIRST_DIM_SIZE`,e[e.VALUE_ROWIDS=1]=`VALUE_ROWIDS`,e[e.ROW_LENGTHS=2]=`ROW_LENGTHS`,e[e.ROW_SPLITS=3]=`ROW_SPLITS`,e[e.ROW_LIMITS=4]=`ROW_LIMITS`,e[e.ROW_STARTS=5]=`ROW_STARTS`})(Oy||={});function ky(e,t,n){let r=[];if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let i=1;i<n.length;++i){let a=n[i],o=r[r.length-n.length+i],s=r[o];if(a>=0){if(s>=0){if(s!==a)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${i+e}] = ${a} but shape[${i+e}] = ${s}`)}else r[o]=a}}return r}function Ay(e){let t={FIRST_DIM_SIZE:Oy.FIRST_DIM_SIZE,VALUE_ROWIDS:Oy.VALUE_ROWIDS,ROW_LENGTHS:Oy.ROW_LENGTHS,ROW_SPLITS:Oy.ROW_SPLITS,ROW_LIMITS:Oy.ROW_LIMITS,ROW_STARTS:Oy.ROW_STARTS},n=[];for(let r of e)if(r in t)n.push(t[r]);else break;return n}function jy(e){return e.length===0?0:e[0]===Oy.FIRST_DIM_SIZE?e.length-1:e.length}function My(e,t){if(e==null||t==null)return;let n=e.length,r=t.length;if(n>=r)throw Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let i=0;i<Math.min(n,r-1);++i){let n=e[i],r=t[i+1];if(n>=0&&r>=0&&n!==1&&n!==r)throw Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${i-e.length}] = ${n} but ragged tensor input.flatValues.shape[${i-e.length}] = ${r}`)}}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
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
function Ny(e){return e<=30?e:Se(e,Math.floor(Math.sqrt(e)))}
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
function Py(e,t,n){return[n*(typeof e==`number`?e:e[0]),t*(typeof e==`number`?e:e[1])]}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
function Fy(e,t,n,r=!0){let i=[];if(r)i=i.concat(t.slice(0)),i.push(e[0]/n),i=i.concat(e.slice(1));else{i=i.concat(e[0]);let n=t.length;for(let r=0;r<n;++r)i=i.concat([e[r+1]/t[r],t[r]]);i=i.concat(e.slice(n+1))}return i}function Iy(e,t,n=!0){let r=[];if(n){r.push(t);for(let n=t+1;n<e;++n)n<=2*t?(r.push(n),r.push(n-(t+1))):r.push(n)}else{let n=[],i=[];for(let r=1;r<e;++r)r>=t*2+1||r%2==1?i.push(r):n.push(r);r.push(...n),r.push(0),r.push(...i)}return r}function Ly(e,t,n,r=!0){let i=[];r?i.push(e[0]/n):i.push(e[0]*n);for(let n=1;n<e.length;++n)n<=t.length?r?i.push(t[n-1]*e[n]):i.push(e[n]/t[n-1]):i.push(e[n]);return i}function Ry(e,t){let n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function zy(e,t,n){let r=e.slice(0,1);for(let i=0;i<n;++i)r.push(e[i+1]-t[i][0]-t[i][1]);return r}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
* Copyright 2018 Google LLC. All Rights Reserved.
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
* Copyright 2018 Google LLC. All Rights Reserved.
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
function By(e,t){if(e.length!==t.length)throw Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);let n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function Vy(e){let t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function Hy(e){let t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=0;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Uy(e){let t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=2;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Wy(e,t){return{real:e[t*2],imag:e[t*2+1]}}function Gy(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function Ky(e,t){let n=new Float32Array(e/2),r=new Float32Array(e/2);for(let i=0;i<Math.ceil(e/2);i++){let a=(t?2:-2)*Math.PI*(i/e);n[i]=Math.cos(a),r[i]=Math.sin(a)}return{real:n,imag:r}}function qy(e,t,n){let r=(n?2:-2)*Math.PI*(e/t);return{real:Math.cos(r),imag:Math.sin(r)}}
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
const Jy=/->/g;function Yy(e,t){e=e.replace(/\s/g,``);let n=(e.length-e.replace(Jy,``).length)/2;if(n<1)throw Error(`Equations without an arrow are not supported.`);if(n>1)throw Error(`Equation must contain exactly one arrow ("->").`);let[r,i]=e.split(`->`);x(r.indexOf(`...`)===-1,()=>`The ellipsis notation ("...") is not supported yet.`);let a=r.split(`,`),o=a.length;if(t!==o)throw Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw Error(`Support for more than 2 input tensors is not implemented yet.`);let s=[];for(let e=0;e<i.length;++e){let t=i[e];if(!a.some(e=>e.indexOf(t)!==-1))throw Error(`Output subscripts contain the label ${t} not present in the input subscripts.`);s.indexOf(t)===-1&&s.push(t)}for(let e=0;e<r.length;++e){let t=r[e];s.indexOf(t)===-1&&t!==`,`&&s.push(t)}let c=Array(a.length);for(let e=0;e<o;++e){if(new Set(a[e].split(``)).size!==a[e].length)throw Error(`Found duplicate axes in input component ${a[e]}. Support for duplicate axes in input is not implemented yet.`);c[e]=[];for(let t=0;t<a[e].length;++t)c[e].push(s.indexOf(a[e][t]))}let l=s.length,u=i.length,d=[];for(let e=u;e<l;++e)d.push(e);return{allDims:s,summedDims:d,idDims:c}}function Xy(e,t){let n=Array(e);n.fill(-1);for(let e=0;e<t.length;++e)n[t[e]]=e;let r=[];for(let t=0;t<e;++t)n[t]===-1&&r.push(t);return n=n.filter(e=>e!==-1),{permutationIndices:n,expandDims:r}}function Zy(e,t,n){let r=Array(e);for(let e=0;e<n.length;++e){let i=n[e].shape;for(let n=0;n<t[e].length;++n)r[t[e][n]]===void 0?r[t[e][n]]=i[n]:x(r[t[e][n]]===i[n],()=>`Expected dimension ${r[t[e][n]]} at axis ${n} of input shaped ${JSON.stringify(i)}, but got dimension ${i[n]}`)}}function Qy(e,t){let n=e,r=[],i=0;e.length===0&&n.push(-1),i=e.length+1;for(let e=0;e<i;++e)r.push([]);let a=[];for(let e=0;e<n.length;++e){let i=n[e],o=eb(t,i);for(let t of o)a.indexOf(t)===-1&&(r[e].push(t),a.push(t))}return{path:n,steps:r}}function $y(e){return e.every((e,t)=>e===t)}function eb(e,t){let n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function tb(e,t,n=0){let r=[];if(typeof t==`number`)x(e.shape[n]%t===0,()=>`Number of splits must evenly divide the axis.`),r=Array(t).fill(e.shape[n]/t);else{x(t.reduce((e,t)=>(t===-1&&(e+=1),e),0)<=1,()=>`There should be only one negative value in split array.`);let i=t.indexOf(-1);if(i!==-1){let r=t.reduce((e,t)=>t>0?e+t:e);t[i]=e.shape[n]-r}x(e.shape[n]===t.reduce((e,t)=>e+t),()=>`The sum of sizes must match the size of the axis dimension.`),r=t}return r}
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
function nb(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function rb(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function ib(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}
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
function ab(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function ob(e,t){return`size ${e} must be non-negative, not ${t}`}function sb(){return`reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero`}function cb(e,t){return`Input to reshape is a SparseTensor with ${w(e)}
  dense values, but the requested shape requires a multiple of ${w(t)}. inputShape=${e} outputShape= ${t}`}function lb(e,t){return`Input to reshape is a tensor with ${w(e)} dense values, but the requested shape has ${w(t)}. inputShape=${e} outputShape=${t}`}
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
function ub(){return`segment ids must be >= 0`}function db(){return`segment ids are not increasing`}function fb(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function pb(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var mb=/* @__PURE__ */ s({collectGatherOpShapeInfo:()=>_b,computeOutShape:()=>gb,segOpComputeOptimalWindowSize:()=>hb});function hb(e,t){let n=!1,r;for(e<=30?(r=e,n=!0):r=Se(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=Se(e,r+1);return r}function gb(e,t,n){let r=[],i=e.length;for(let a=0;a<i;a++)a===t?r.push(n):r.push(e[a]);return r}function _b(e,t,n,r){let i=t.shape.length,a=e.shape.length;if(r!==0&&(r<-i||r>i))throw Error(`Expect batchDims in the range of [-${i}, ${i}], but got ${r}`);if(r<0&&(r+=i),r>a)throw Error(`batchDims (${r}) must be less than rank(x) (
    ${a}).`);if(n<r)throw Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let n=0;n<r;++n)if(e.shape[n]!==t.shape[n])throw Error(`x.shape[${n}]: ${e.shape[n]} should be equal to indices.shape[${n}]: ${t.shape[n]}.`);let o=e.shape[n],s=[],c=1,l=1,u=1;for(let t=0;t<r;++t)s.push(e.shape[t]),c*=e.shape[t];for(let t=r;t<n;t++)s.push(e.shape[t]),l*=e.shape[t];for(let e=r;e<i;e++)s.push(t.shape[e]);for(let t=n+1;t<a;t++)s.push(e.shape[t]),u*=e.shape[t];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:o,outputShape:s}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
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
var vb=/* @__PURE__ */ s({ERF_A1:()=>.254829592,ERF_A2:()=>-.284496736,ERF_A3:()=>1.421413741,ERF_A4:()=>-1.453152027,ERF_A5:()=>1.061405429,ERF_P:()=>.3275911,PARALLELIZE_THRESHOLD:()=>30,RowPartitionType:()=>Oy,SELU_SCALE:()=>1.0507009873554805,SELU_SCALEALPHA:()=>1.7580993408473768,applyActivation:()=>Gh,assertAndGetBroadcastShape:()=>Y,assertAxesAreInnerMostDims:()=>du,assertParamsConsistent:()=>Ey,assignToTypedArray:()=>Gy,axesAreInnerMostDims:()=>su,calculateShapes:()=>rh,checkEinsumDimSizes:()=>Zy,checkPadOnDimRoundingMode:()=>hc,combineLocations:()=>cu,combineRaggedTensorToTensorShapes:()=>ky,complexWithEvenIndex:()=>Hy,complexWithOddIndex:()=>Uy,computeConv2DInfo:()=>ec,computeConv3DInfo:()=>tc,computeDefaultPad:()=>ic,computeDilation2DInfo:()=>Zs,computeOptimalWindowSize:()=>Ny,computeOutAndReduceShapes:()=>lu,computeOutShape:()=>Dy,computePool2DInfo:()=>Qs,computePool3DInfo:()=>$s,convertConv2DDataFormat:()=>mc,decodeEinsumEquation:()=>Yy,eitherStridesOrDilationsAreOne:()=>fc,expandShapeToKeepDim:()=>uu,exponent:()=>qy,exponents:()=>Ky,fromStringArrayToUint8:()=>bb,fromUint8ToStringArray:()=>yb,getAxesPermutation:()=>fu,getBroadcastDims:()=>Vl,getComplexWithIndex:()=>Wy,getEinsumComputePath:()=>Qy,getEinsumPermutation:()=>Xy,getFusedBiasGradient:()=>Wh,getFusedDyActivation:()=>Uh,getImageCenter:()=>Py,getInnerMostAxes:()=>mu,getPermuted:()=>Iy,getRaggedRank:()=>jy,getReductionAxes:()=>Hl,getReshaped:()=>Fy,getReshapedPermuted:()=>Ly,getRowPartitionTypesHelper:()=>Ay,getSliceBeginCoords:()=>Ry,getSliceSize:()=>zy,getSparseFillEmptyRowsIndicesDenseShapeMismatch:()=>nb,getSparseFillEmptyRowsNegativeIndexErrorMessage:()=>rb,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:()=>ib,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:()=>sb,getSparseReshapeInputOutputMismatchErrorMessage:()=>lb,getSparseReshapeInputOutputMultipleErrorMessage:()=>cb,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:()=>ab,getSparseReshapeNegativeOutputDimErrorMessage:()=>ob,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:()=>pb,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:()=>ub,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:()=>db,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:()=>fb,getUndoAxesPermutation:()=>pu,isIdentityPermutation:()=>$y,log:()=>ii,mergeRealAndImagArrays:()=>By,prepareAndValidate:()=>ny,prepareSplitSize:()=>tb,segment_util:()=>mb,shouldFuse:()=>Kh,slice_util:()=>ry,splitRealAndImagArrays:()=>Vy,stridesOrDilationsArePositive:()=>pc,tupleValuesAreOne:()=>dc,upcastType:()=>da,validateDefaultValueShape:()=>My,validateInput:()=>nh,validateUpdateShape:()=>th,warn:()=>ri});function yb(e){try{return e.map(e=>zi(e))}catch(e){throw Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function bb(e){return e.map(e=>Ri(e))}var xb=/* @__PURE__ */ s({nonMaxSuppressionV3Impl:()=>kg,nonMaxSuppressionV4Impl:()=>Ag,nonMaxSuppressionV5Impl:()=>jg,whereImpl:()=>vh}),Sb=/* @__PURE__ */ s({Abs:()=>`Abs`,Acos:()=>Ke,Acosh:()=>qe,AdadeltaOptimizer:()=>av,AdagradOptimizer:()=>ov,AdamOptimizer:()=>sv,AdamaxOptimizer:()=>cv,Add:()=>`Add`,AddN:()=>Ye,All:()=>`All`,Any:()=>`Any`,ArgMax:()=>Qe,ArgMin:()=>$e,Asin:()=>et,Asinh:()=>tt,Atan:()=>nt,Atan2:()=>it,Atanh:()=>rt,AvgPool:()=>at,AvgPool3D:()=>st,AvgPool3DGrad:()=>ct,AvgPoolGrad:()=>ot,BatchMatMul:()=>lt,BatchToSpaceND:()=>ut,Bincount:()=>dt,BitwiseAnd:()=>ft,BroadcastArgs:()=>pt,BroadcastTo:()=>`BroadcastTo`,Cast:()=>mt,Ceil:()=>ht,ClipByValue:()=>gt,Complex:()=>_t,ComplexAbs:()=>vt,Concat:()=>yt,Conv2D:()=>bt,Conv2DBackpropFilter:()=>xt,Conv2DBackpropInput:()=>St,Conv3D:()=>Ct,Conv3DBackpropFilterV2:()=>wt,Conv3DBackpropInputV2:()=>Tt,Cos:()=>`Cos`,Cosh:()=>Dt,CropAndResize:()=>At,Cumprod:()=>Ot,Cumsum:()=>kt,DataStorage:()=>u,DenseBincount:()=>jt,DepthToSpace:()=>Mt,DepthwiseConv2dNative:()=>Nt,DepthwiseConv2dNativeBackpropFilter:()=>Pt,DepthwiseConv2dNativeBackpropInput:()=>Ft,Diag:()=>It,Dilation2D:()=>Lt,Dilation2DBackpropFilter:()=>zt,Dilation2DBackpropInput:()=>Rt,Draw:()=>Bt,ENV:()=>ze,Einsum:()=>Ht,Elu:()=>`Elu`,EluGrad:()=>Wt,Environment:()=>Fe,Equal:()=>Kt,Erf:()=>`Erf`,Exp:()=>`Exp`,ExpandDims:()=>Jt,Expm1:()=>Yt,FFT:()=>`FFT`,Fill:()=>Xt,FlipLeftRight:()=>Zt,Floor:()=>Qt,FloorDiv:()=>$t,FromPixels:()=>Qr,FusedBatchNorm:()=>en,FusedConv2D:()=>ti,FusedDepthwiseConv2D:()=>ni,GatherNd:()=>nn,GatherV2:()=>tn,Greater:()=>rn,GreaterEqual:()=>an,IFFT:()=>sn,Identity:()=>on,Imag:()=>cn,IsFinite:()=>ln,IsInf:()=>un,IsNan:()=>dn,KernelBackend:()=>d,LRN:()=>`LRN`,LRNGrad:()=>Cn,LeakyRelu:()=>fn,Less:()=>pn,LessEqual:()=>mn,LinSpace:()=>hn,Log:()=>`Log`,Log1p:()=>_n,LogSoftmax:()=>`LogSoftmax`,LogicalAnd:()=>vn,LogicalNot:()=>yn,LogicalOr:()=>bn,LogicalXor:()=>xn,LowerBound:()=>`LowerBound`,MatrixBandPart:()=>`MatrixBandPart`,Max:()=>`Max`,MaxPool:()=>En,MaxPool3D:()=>On,MaxPool3DGrad:()=>kn,MaxPoolGrad:()=>Dn,MaxPoolWithArgmax:()=>An,Maximum:()=>Tn,Mean:()=>jn,Min:()=>`Min`,Minimum:()=>Nn,MirrorPad:()=>Pn,Mod:()=>`Mod`,MomentumOptimizer:()=>uv,Multinomial:()=>In,Multiply:()=>Ln,Neg:()=>`Neg`,NonMaxSuppressionV3:()=>Bn,NonMaxSuppressionV4:()=>Vn,NonMaxSuppressionV5:()=>Hn,NotEqual:()=>zn,OP_SCOPE_SUFFIX:()=>La,OneHot:()=>Wn,OnesLike:()=>Un,Optimizer:()=>iv,OptimizerConstructors:()=>Sy,Pack:()=>Gn,PadV2:()=>Kn,Pool:()=>`Pool`,Pow:()=>`Pow`,Prelu:()=>Jn,Prod:()=>Yn,RMSPropOptimizer:()=>dv,RaggedGather:()=>Xn,RaggedRange:()=>Zn,RaggedTensorToTensor:()=>Qn,Range:()=>$n,Rank:()=>aa,Real:()=>er,RealDiv:()=>Vt,Reciprocal:()=>tr,Reduction:()=>$,Relu:()=>nr,Relu6:()=>cr,Reshape:()=>rr,ResizeBilinear:()=>or,ResizeBilinearGrad:()=>sr,ResizeNearestNeighbor:()=>ir,ResizeNearestNeighborGrad:()=>ar,Reverse:()=>lr,RotateWithOffset:()=>$r,Round:()=>ur,Rsqrt:()=>dr,SGDOptimizer:()=>lv,ScatterNd:()=>fr,SearchSorted:()=>mr,Select:()=>hr,Selu:()=>gr,Sigmoid:()=>xr,Sign:()=>br,Sin:()=>`Sin`,Sinh:()=>yr,Slice:()=>_r,Softmax:()=>Dr,Softplus:()=>Sr,SpaceToBatchND:()=>Tr,SparseFillEmptyRows:()=>Or,SparseReshape:()=>kr,SparseSegmentMean:()=>Ar,SparseSegmentSum:()=>jr,SparseToDense:()=>Mr,SplitV:()=>Er,Sqrt:()=>Cr,Square:()=>Pr,SquaredDifference:()=>Nr,StaticRegexReplace:()=>Fr,Step:()=>Zr,StridedSlice:()=>Ir,StringNGrams:()=>Lr,StringSplit:()=>Rr,StringToHashBucketFast:()=>zr,Sub:()=>`Sub`,Sum:()=>`Sum`,Tan:()=>`Tan`,Tanh:()=>Hr,Tensor:()=>M,TensorBuffer:()=>Qi,TensorScatterUpdate:()=>pr,Tile:()=>Ur,TopK:()=>Wr,Transform:()=>Gr,Transpose:()=>Kr,Unique:()=>qr,Unpack:()=>Jr,UnsortedSegmentSum:()=>Yr,UpperBound:()=>`UpperBound`,Variable:()=>ia,ZerosLike:()=>Xr,_FusedMatMul:()=>ei,abs:()=>W,acos:()=>Os,acosh:()=>As,add:()=>V,addN:()=>Ms,all:()=>Ps,any:()=>Is,argMax:()=>Rs,argMin:()=>Bs,asin:()=>Hs,asinh:()=>Ws,atan:()=>Ks,atan2:()=>Js,atanh:()=>Xs,avgPool:()=>vc,avgPool3d:()=>bc,backend:()=>co,backend_util:()=>vb,basicLSTMCell:()=>kc,batchNorm:()=>Pc,batchNorm2d:()=>Ic,batchNorm3d:()=>Rc,batchNorm4d:()=>Bc,batchToSpaceND:()=>jc,bincount:()=>Hc,bitwiseAnd:()=>Wc,booleanMaskAsync:()=>Sh,broadcastArgs:()=>Kc,broadcastTo:()=>Jc,broadcast_util:()=>Bl,browser:()=>Vv,buffer:()=>gs,cast:()=>B,ceil:()=>Xc,clipByValue:()=>$c,clone:()=>ys,complex:()=>za,concat:()=>K,concat1d:()=>tl,concat2d:()=>rl,concat3d:()=>al,concat4d:()=>sl,conv1d:()=>dl,conv2d:()=>ll,conv2dTranspose:()=>hl,conv3d:()=>_l,conv3dTranspose:()=>xl,copyRegisteredKernels:()=>mi,cos:()=>Cl,cosh:()=>Tl,cosineWindow:()=>Rh,cumprod:()=>Dl,cumsum:()=>kl,customGrad:()=>xd,denseBincount:()=>jl,deprecationWarn:()=>Ja,depthToSpace:()=>Nl,depthwiseConv2d:()=>Fl,device_util:()=>Da,diag:()=>Ll,dilation2d:()=>zl,disableDeprecationWarnings:()=>qa,dispose:()=>z,disposeVariables:()=>Ya,div:()=>H,divNoNan:()=>Xl,dot:()=>Ql,dropout:()=>Ih,einsum:()=>eu,elu:()=>nu,enableDebugMode:()=>Ka,enableProdMode:()=>Ga,enclosingPowerOfTwo:()=>Lh,engine:()=>Xa,ensureShape:()=>iu,env:()=>D,equal:()=>Wl,erf:()=>ou,euclideanNorm:()=>Au,exp:()=>Mu,expandDims:()=>Pu,expm1:()=>Iu,eye:()=>Bu,fft:()=>Dm,fill:()=>Zc,findBackend:()=>ao,findBackendFactory:()=>oo,floor:()=>Hu,floorDiv:()=>Cs,fused:()=>rg,gather:()=>Wu,gatherND:()=>Nh,gather_util:()=>ty,getBackend:()=>ro,getGradient:()=>ci,getKernel:()=>si,getKernelsForBackend:()=>li,grad:()=>gd,grads:()=>_d,greater:()=>Ku,greaterEqual:()=>Ju,ifft:()=>km,imag:()=>Xu,image:()=>K_,inTopKAsync:()=>Bh,io:()=>Lv,irfft:()=>jm,isFinite:()=>Qu,isInf:()=>ed,isNaN:()=>nd,keep:()=>$a,kernel_impls:()=>xb,leakyRelu:()=>id,less:()=>od,lessEqual:()=>cd,linalg:()=>q_,linspace:()=>ld,localResponseNormalization:()=>dd,log:()=>pd,log1p:()=>hd,logSigmoid:()=>Od,logSoftmax:()=>jd,logSumExp:()=>Nd,logicalAnd:()=>Fd,logicalNot:()=>Ld,logicalOr:()=>zd,logicalXor:()=>Vd,losses:()=>J_,lowerBound:()=>Gd,matMul:()=>q,math:()=>Bv,max:()=>gu,maxPool:()=>qd,maxPool3d:()=>Yd,maxPoolWithArgmax:()=>Zd,maximum:()=>$d,mean:()=>tf,memory:()=>Za,meshgrid:()=>af,min:()=>vu,minimum:()=>sf,mirrorPad:()=>lf,mod:()=>df,moments:()=>pf,movingAverage:()=>Eh,mul:()=>U,multiRNNCell:()=>hf,multinomial:()=>_f,neg:()=>wd,nextFrame:()=>Ty,norm:()=>Ou,notEqual:()=>yf,oneHot:()=>xf,ones:()=>rf,onesLike:()=>Cf,op:()=>L,outerProduct:()=>Tf,pad:()=>Df,pad1d:()=>kf,pad2d:()=>jf,pad3d:()=>Nf,pad4d:()=>Ff,pool:()=>Vf,pow:()=>bu,prelu:()=>Uf,print:()=>bs,prod:()=>Gf,profile:()=>Qa,raggedGather:()=>qf,raggedRange:()=>Yf,raggedTensorToTensor:()=>Zf,rand:()=>$f,randomGamma:()=>Ep,randomNormal:()=>Op,randomStandardNormal:()=>Ap,randomUniform:()=>Mp,randomUniformInt:()=>Pp,range:()=>Fp,ready:()=>no,real:()=>Lp,reciprocal:()=>zp,registerBackend:()=>so,registerGradient:()=>di,registerKernel:()=>ui,relu:()=>Vp,relu6:()=>Up,removeBackend:()=>io,reshape:()=>G,reverse:()=>Gp,reverse1d:()=>qp,reverse2d:()=>Yp,reverse3d:()=>Zp,reverse4d:()=>$p,rfft:()=>Fm,round:()=>tm,rsqrt:()=>rm,scalar:()=>X,scatterND:()=>Oh,scatter_util:()=>eh,searchSorted:()=>Wd,selu:()=>am,separableConv2d:()=>sm,serialization:()=>Z_,setBackend:()=>to,setPlatform:()=>lo,setdiff1dAsync:()=>lm,sigmoid:()=>wc,sign:()=>dm,signal:()=>G_,sin:()=>pm,sinh:()=>hm,slice:()=>J,slice1d:()=>_m,slice2d:()=>ym,slice3d:()=>xm,slice4d:()=>Cm,slice_util:()=>ry,softmax:()=>Tm,softplus:()=>Ed,spaceToBatchND:()=>Lf,sparse:()=>Y_,sparseToDense:()=>jh,spectral:()=>W_,split:()=>Nm,sqrt:()=>Su,square:()=>wu,squaredDifference:()=>Lm,squeeze:()=>zm,stack:()=>Vm,step:()=>Um,stridedSlice:()=>Gm,string:()=>X_,sub:()=>Q,sum:()=>Z,sumOutType:()=>fa,tan:()=>qm,tanh:()=>Dc,tensor:()=>Va,tensor1d:()=>Jm,tensor2d:()=>Ym,tensor3d:()=>Xm,tensor4d:()=>Zm,tensor5d:()=>Qm,tensor6d:()=>$m,tensorScatterUpdate:()=>ah,tensor_util:()=>ha,test_util:()=>cp,tidy:()=>R,tile:()=>Ru,time:()=>eo,topk:()=>sh,train:()=>Cy,transpose:()=>wh,truncatedNormal:()=>lh,unique:()=>dh,unregisterGradient:()=>pi,unregisterKernel:()=>fi,unsortedSegmentSum:()=>ph,unstack:()=>hh,upcastType:()=>da,upperBound:()=>gh,util:()=>Mi,valueAndGrad:()=>vd,valueAndGrads:()=>yd,variable:()=>_h,variableGrads:()=>bd,version_core:()=>`4.22.0`,where:()=>Kl,whereAsync:()=>bh,zeros:()=>nf,zerosLike:()=>Jl})
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
* Copyright 2017 Google LLC. All Rights Reserved.
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
pv();export{_h as $,It as $a,$a as $i,Ju as $n,Vn as $o,Xc as $r,pr as $s,$f as $t,yy as A,ot as Aa,Bs as Ai,Gd as An,hn as Ao,Kl as Ar,yr as As,hm as At,X_ as B,bt as Ba,Cs as Bi,Ed as Bn,kn as Bo,Tl as Br,Er as Bs,Yp as Bt,Py as C,tt as Ca,d as Cc,mc as Ci,rf as Cn,un as Co,ou as Cr,fr as Cs,km as Ct,oy as D,at as Da,Ks as Di,Zd as Dn,fn as Do,Ql as Dr,xr as Ds,xm as Dt,_y as E,rt as Ea,l as Ec,Js as Ei,$d as En,Cn as Eo,eu as Er,gr as Es,Cm as Et,q_ as F,pt as Fa,As as Fi,Fd as Fn,bn as Fo,Fl as Fr,Or as Fs,am as Ft,Ih as G,Et as Ga,gs as Gi,ld as Gn,Mn as Go,dl as Gr,Ir as Gs,zp as Gt,Bh as H,Ct as Ha,bs as Hi,hd as Hn,An as Ho,xl as Hr,Pr as Hs,Gp as Ht,J_ as I,mt as Ia,Os as Ii,Nd as In,xn as Io,Nl as Ir,kr as Is,rm as It,Oh as J,Ot as Ja,Eo as Ji,id as Jn,Fn as Jo,al as Jr,zr as Js,Pp as Jt,Nh as K,Dt as Ka,fs as Ki,cd as Kn,Nn as Ko,ll as Kr,Lr as Ks,Lp as Kt,G_ as L,ht as La,W as Li,jd as Ln,wn as Lo,jl as Lr,Ar as Ls,tm as Lt,Lv as M,ut as Ma,Is as Mi,Vd as Mn,_n as Mo,Y as Mr,Dr as Ms,dm as Mt,Pv as N,dt as Na,Ps as Ni,zd as Nn,vn as No,zl as Nr,Sr as Ns,lm as Nt,gy as O,st as Oa,Ws as Oi,Yd as On,pn as Oo,Xl as Or,br as Os,ym as Ot,K_ as P,ft as Pa,Ms as Pi,Ld as Pn,yn as Po,Ll as Pr,Tr as Ps,sm as Pt,bh as Q,Nt as Qa,Xa as Qi,Xu as Qn,Bn as Qo,Zc as Qr,Hr as Qs,Ep as Qt,Y_ as R,gt as Ra,U as Ri,Q as Rn,En as Ro,kl as Rr,jr as Rs,$p as Rt,zy as S,et as Sa,u as Sc,$s as Si,af as Sn,ln as So,pu as Sr,dr as Ss,jm as St,Dy as T,it as Ta,s as Tc,Xs as Ti,tf as Tn,Sn as To,nu as Tr,hr as Ts,Tm as Tt,Rh as U,wt as Ua,ys as Ui,pd as Un,Tn as Uo,_l as Ur,Nr as Us,Up as Ut,rg as V,St as Va,V as Vi,wd as Vn,Dn as Vo,Cl as Vr,Cr as Vs,qp as Vt,Lh as W,Tt as Wa,B as Wi,dd as Wn,jn as Wo,hl as Wr,Zr as Ws,Vp as Wt,wh as X,jt as Xa,Ja as Xi,ed as Xn,Ln as Xo,tl as Xr,wr as Xs,Ap as Xt,Eh as Y,kt as Ya,ko as Yi,nd as Yn,In as Yo,rl as Yr,Br as Ys,Mp as Yt,Sh as Z,Mt as Za,z as Zi,Qu as Zn,Rn as Zo,$c as Zr,Vr as Zs,Op as Zt,tb as _,Ye as _a,Ne as _c,G as _i,hf as _n,nn as _o,du as _r,ir as _s,Vm as _t,pb as a,L as aa,Jr as ac,Rc as ai,Vf as an,Kt as ao,Iu as ar,Kn as as,sh as at,Ly as b,Qe as ba,w as bc,Zs as bi,lf as bn,an as bo,fu as br,$r as bs,Fm as bt,fb as c,Ri as ca,D as cc,jc as ci,Nf as cn,Jt as co,Au as cr,Yn as cs,$m as ct,cb as d,ji as da,S as dc,J as di,Df as dn,Zt as do,wu as dr,tr as ds,Xm as dt,so as ea,Ur as ec,Jc as ei,Zf as en,Lt as eo,Ku as er,Hn as es,gh as et,ab as f,ui as fa,he as fc,wc as fi,Tf as fn,Qt as fo,Su as fr,nr as fs,Ym as ft,ib as g,Je as ga,se as gc,vc as gi,_f as gn,ni as go,gu as gr,sr as gs,Um as gt,rb as h,qe as ha,be as hc,bc as hi,yf as hn,ti as ho,vu as hr,or as hs,Gm as ht,_b as i,La as ia,qr as ic,Bc as ii,Uf as in,Wt as io,Ru as ir,Gn as is,lh as it,ny as j,lt as ja,Rs as ji,Wd as jn,gn as jo,Wl as jr,_r as js,pm as jt,vy as k,ct as ka,Hs as ki,qd as kn,mn as ko,Jl as kr,vr as ks,_m as kt,sb as l,Li as la,E as lc,kc as li,jf as ln,Yt as lo,Ou as lr,$n as ls,Qm as lt,nb as m,Ke as ma,de as mc,K as mi,xf as mn,en as mo,bu as mr,rr as ms,qm as mt,bb as n,Va as na,Gr as nc,Wc as ni,qf as nn,Rt as no,Hu as nr,Wn as ns,ph as nt,ub as o,M as oa,Xr as oc,Ic as oi,Lf as on,Gt as oo,Pu as or,qn as os,ah as ot,ob as p,Ge as pa,Ce as pc,q as pi,Cf as pn,$t as po,X as pr,cr as ps,Jm as pt,jh as q,At as qa,_o as qi,od as qn,Pn as qo,sl as qr,Rr as qs,Fp as qt,yb as r,za as ra,Kr as rc,Hc as ri,Gf as rn,Ut as ro,Bu as rr,Un as rs,dh as rt,db as s,Qi as sa,ei as sc,Pc as si,Ff as sn,qt as so,Mu as sr,Jn as ss,rh as st,Sb as t,R as ta,Wr as tc,Kc as ti,Yf as tn,zt as to,Wu as tr,zn as ts,hh as tt,lb as u,Ii as ua,x as uc,Dc as ui,kf as un,Xt as uo,Z as ur,Vt as us,Zm as ut,Iy as v,Xe as va,Oe as vc,ec as vi,pf as vn,tn as vo,lu as vr,ar as vs,zm as vt,Ey as w,nt as wa,o as wc,fc as wi,nf as wn,dn as wo,iu as wr,mr as ws,Dm as wt,Ry as x,$e as xa,Te as xc,Qs as xi,sf as xn,on as xo,mu as xr,ur as xs,Nm as xt,Fy as y,Ze as ya,ce as yc,tc as yi,df as yn,rn as yo,uu as yr,lr as ys,Lm as yt,W_ as z,yt as za,H as zi,Od as zn,On as zo,Dl as zr,Mr as zs,Zp as zt};
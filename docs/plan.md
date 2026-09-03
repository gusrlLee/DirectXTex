# Plan

## Evaluation Metrics
* Quality 
  * PSNR: texdiag
  * FLIP: python
* Performance 
  * second: texconv -timing options
 
* pure read/write time 은 측정에서 제외 
* input 은 compressed mip0 dds file 고정

## baseline 

write time 은 시간 빼기 

read / downsampling / [encoding] / write 

(나중에 다시)
* etcpak 2.1 
* AMD Compressonator
* Microsoft DirectXTex 

## Feature list 
* [BC1](./BC1.md)
* BC2
* BC3
* [BC4](./BC4.md)
* BC5
* [BC7](./BC7.md)
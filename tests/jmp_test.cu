// Large dense switch -> jump table, hoping for JMP (const/imm) or BRX.
__global__ void bigswitch(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    int r;
    switch (v & 31) {
        case 0:  r = v + 1;   break;   case 1:  r = v * 2;   break;
        case 2:  r = v - 3;   break;   case 3:  r = v ^ 4;   break;
        case 4:  r = v << 1;  break;   case 5:  r = v >> 2;  break;
        case 6:  r = v * v;   break;   case 7:  r = v + 100; break;
        case 8:  r = v - 100; break;   case 9:  r = v | 8;   break;
        case 10: r = v & 15;  break;   case 11: r = v * 7;   break;
        case 12: r = v + 21;  break;   case 13: r = v - 42;  break;
        case 14: r = v * 3+1; break;   case 15: r = v / 3;   break;
        case 16: r = v % 7;   break;   case 17: r = v + 55;  break;
        case 18: r = ~v;      break;   case 19: r = -v;      break;
        case 20: r = v << 3;  break;   case 21: r = v >> 1;  break;
        case 22: r = v * 11;  break;   case 23: r = v + 9;   break;
        case 24: r = v - 9;   break;   case 25: r = v ^ 0x55;break;
        case 26: r = v + 256; break;   case 27: r = v * 13;  break;
        case 28: r = v & 0xff;break;   case 29: r = v | 0xf00;break;
        case 30: r = v * v+v; break;   default: r = 0;       break;
    }
    out[i] = r;
}

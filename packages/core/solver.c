typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned int size_t;
#define NULL ((void*)0)

#define memcpy __builtin_memcpy
#define memset __builtin_memset
#define memcmp __builtin_memcmp

#define RATE 136

static const uint64_t RC_X[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

static inline uint64_t rotl64_x(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

void keccak_f1600(uint64_t s[25], int start, int end) {
    for (int r = start; r < end; r++) {
        uint64_t c0 = s[0]^s[5]^s[10]^s[15]^s[20];
        uint64_t c1 = s[1]^s[6]^s[11]^s[16]^s[21];
        uint64_t c2 = s[2]^s[7]^s[12]^s[17]^s[22];
        uint64_t c3 = s[3]^s[8]^s[13]^s[18]^s[23];
        uint64_t c4 = s[4]^s[9]^s[14]^s[19]^s[24];
        uint64_t d0 = c4 ^ rotl64_x(c1,1), d1 = c0 ^ rotl64_x(c2,1), d2 = c1 ^ rotl64_x(c3,1);
        uint64_t d3 = c2 ^ rotl64_x(c4,1), d4 = c3 ^ rotl64_x(c0,1);
        s[0]^=d0;s[1]^=d1;s[2]^=d2;s[3]^=d3;s[4]^=d4;
        s[5]^=d0;s[6]^=d1;s[7]^=d2;s[8]^=d3;s[9]^=d4;
        s[10]^=d0;s[11]^=d1;s[12]^=d2;s[13]^=d3;s[14]^=d4;
        s[15]^=d0;s[16]^=d1;s[17]^=d2;s[18]^=d3;s[19]^=d4;
        s[20]^=d0;s[21]^=d1;s[22]^=d2;s[23]^=d3;s[24]^=d4;

        uint64_t cur = s[1], tmp;
        tmp=s[10]; s[10]=rotl64_x(cur,1);  cur=tmp;
        tmp=s[7];  s[7] =rotl64_x(cur,3);  cur=tmp;
        tmp=s[11]; s[11]=rotl64_x(cur,6);  cur=tmp;
        tmp=s[17]; s[17]=rotl64_x(cur,10); cur=tmp;
        tmp=s[18]; s[18]=rotl64_x(cur,15); cur=tmp;
        tmp=s[3];  s[3] =rotl64_x(cur,21); cur=tmp;
        tmp=s[5];  s[5] =rotl64_x(cur,28); cur=tmp;
        tmp=s[16]; s[16]=rotl64_x(cur,36); cur=tmp;
        tmp=s[8];  s[8] =rotl64_x(cur,45); cur=tmp;
        tmp=s[21]; s[21]=rotl64_x(cur,55); cur=tmp;
        tmp=s[24]; s[24]=rotl64_x(cur,2);  cur=tmp;
        tmp=s[4];  s[4] =rotl64_x(cur,14); cur=tmp;
        tmp=s[15]; s[15]=rotl64_x(cur,27); cur=tmp;
        tmp=s[23]; s[23]=rotl64_x(cur,41); cur=tmp;
        tmp=s[19]; s[19]=rotl64_x(cur,56); cur=tmp;
        tmp=s[13]; s[13]=rotl64_x(cur,8);  cur=tmp;
        tmp=s[12]; s[12]=rotl64_x(cur,25); cur=tmp;
        tmp=s[2];  s[2] =rotl64_x(cur,43); cur=tmp;
        tmp=s[20]; s[20]=rotl64_x(cur,62); cur=tmp;
        tmp=s[14]; s[14]=rotl64_x(cur,18); cur=tmp;
        tmp=s[22]; s[22]=rotl64_x(cur,39); cur=tmp;
        tmp=s[9];  s[9] =rotl64_x(cur,61); cur=tmp;
        tmp=s[6];  s[6] =rotl64_x(cur,20); cur=tmp;
                   s[1] =rotl64_x(cur,44);

        { uint64_t l0=s[0],l1=s[1],l2=s[2],l3=s[3],l4=s[4];
          s[0]=l0^(~l1&l2); s[1]=l1^(~l2&l3); s[2]=l2^(~l3&l4); s[3]=l3^(~l4&l0); s[4]=l4^(~l0&l1); }
        { uint64_t l0=s[5],l1=s[6],l2=s[7],l3=s[8],l4=s[9];
          s[5]=l0^(~l1&l2); s[6]=l1^(~l2&l3); s[7]=l2^(~l3&l4); s[8]=l3^(~l4&l0); s[9]=l4^(~l0&l1); }
        { uint64_t l0=s[10],l1=s[11],l2=s[12],l3=s[13],l4=s[14];
          s[10]=l0^(~l1&l2); s[11]=l1^(~l2&l3); s[12]=l2^(~l3&l4); s[13]=l3^(~l4&l0); s[14]=l4^(~l0&l1); }
        { uint64_t l0=s[15],l1=s[16],l2=s[17],l3=s[18],l4=s[19];
          s[15]=l0^(~l1&l2); s[16]=l1^(~l2&l3); s[17]=l2^(~l3&l4); s[18]=l3^(~l4&l0); s[19]=l4^(~l0&l1); }
        { uint64_t l0=s[20],l1=s[21],l2=s[22],l3=s[23],l4=s[24];
          s[20]=l0^(~l1&l2); s[21]=l1^(~l2&l3); s[22]=l2^(~l3&l4); s[23]=l3^(~l4&l0); s[24]=l4^(~l0&l1); }

        s[0] ^= RC_X[r];
    }
}

void deepseek_hash(const uint8_t* input, int len, uint8_t out[32]) {
    uint64_t s[25] = {0};
    int k = (RATE - ((len + 2) % RATE)) % RATE;
    int padded = len + 2 + k;
    for (int off = 0; off < padded; off += RATE) {
        for (int j = 0; j < RATE; j++) {
            int pos = off + j;
            uint64_t byte = pos < len ? input[pos] : pos == len ? 0x06 : pos == padded-1 ? 0x80 : 0;
            s[j>>3] ^= byte << ((j&7)<<3);
        }
        keccak_f1600(s, 1, 24);
    }
    for (int i = 0; i < 32; i++) out[i] = (uint8_t)(s[i>>3] >> ((i&7)<<3));
}

static inline int digit_count(int64_t v) {
    int n = 1; v /= 10;
    while (v > 0) { n++; v /= 10; }
    return n;
}

int solve_pow_opt(const uint8_t* salt, int saltLen, int64_t expireAt, int32_t difficulty, const uint8_t* target, int targetLen) {
    if (difficulty < 0) return -1;

    uint8_t prefix[256]; int pl = saltLen;
    for (int i = 0; i < saltLen; i++) prefix[i] = salt[i]; prefix[pl++] = '_';
    {
        char digits[32]; int nb = 0; int64_t t = expireAt;
        if (t < 0) { prefix[pl++] = '-'; t = -t; }
        if (t == 0) { prefix[pl++] = '0'; }
        else { int ds = pl; while (t > 0) { digits[nb++] = '0' + (int)(t%10); t/=10; } while (nb>0) prefix[ds++] = digits[--nb]; pl = ds; }
    }
    prefix[pl++] = '_';

    uint8_t targetBytes[32];
    for (int i = 0; i < 32; i++) {
        char hi = target[i*2], lo = target[i*2+1];
        int hiv = hi>='0'&&hi<='9'?hi-48:hi>='A'&&hi<='F'?hi-55:hi>='a'&&hi<='f'?hi-87:0;
        int lov = lo>='0'&&lo<='9'?lo-48:lo>='A'&&lo<='F'?lo-55:lo>='a'&&lo<='f'?lo-87:0;
        targetBytes[i] = (uint8_t)((hiv<<4) | lov);
    }
    uint64_t tgt0, tgt1, tgt2, tgt3;
    memcpy(&tgt0, targetBytes,      8);
    memcpy(&tgt1, targetBytes + 8,  8);
    memcpy(&tgt2, targetBytes + 16, 8);
    memcpy(&tgt3, targetBytes + 24, 8);

    int maxDigits = digit_count(difficulty);

    if (pl + maxDigits + 2 <= RATE) {
        uint64_t prefixState[25] = {0};
        {
            uint8_t block[RATE]; memset(block, 0, RATE);
            memcpy(block, prefix, pl);
            for (int i = 0; i < RATE/8; i++) memcpy(&prefixState[i], block + 8*i, 8);
        }

        char nd[16]; int ndigits = 1; nd[0] = '0';
        uint64_t s[25];

        for (int32_t n = 0; n <= difficulty; n++) {
            memcpy(s, prefixState, sizeof(prefixState));

            int base = pl;
            for (int j = 0; j < ndigits; j++) {
                int pos = base + j, lane = pos >> 3, sh = (pos & 7) << 3;
                s[lane] ^= (uint64_t)(uint8_t)nd[j] << sh;
            }
            {
                int pos = base + ndigits, lane = pos >> 3, sh = (pos & 7) << 3;
                s[lane] ^= (uint64_t)0x06 << sh;
            }
            {
                int pos = RATE - 1, lane = pos >> 3, sh = (pos & 7) << 3;
                s[lane] ^= (uint64_t)0x80 << sh;
            }

            keccak_f1600(s, 1, 24);

            if (s[0]==tgt0 && s[1]==tgt1 && s[2]==tgt2 && s[3]==tgt3) return n;

            int p = ndigits - 1;
            while (p >= 0) {
                if (nd[p] != '9') { nd[p]++; break; }
                nd[p] = '0'; p--;
            }
            if (p < 0) {
                for (int j = ndigits; j > 0; j--) nd[j] = nd[j-1];
                nd[0] = '1'; ndigits++;
            }
        }
        return -1;
    }

    uint8_t buf[288];
    for (int i = 0; i < pl; i++) buf[i] = prefix[i];
    char digits[32];
    for (int32_t n = 0; n <= difficulty; n++) {
        int no = pl, val = n, nb;
        if (val == 0) { buf[no++] = '0'; }
        else { nb = 0; do { digits[nb++] = '0'+(val%10); val/=10; } while (val>0); while (nb>0) buf[no++] = digits[--nb]; }

        uint8_t hash[32];
        deepseek_hash(buf, no, hash);

        if (memcmp(hash, targetBytes, 32) == 0) return n;
    }
    return -1;
}
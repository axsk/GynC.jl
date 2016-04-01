hplus(s, t, n)  = (s/t)^n / (1. + (s/t)^n)
hminus(s, t, n) = 1.      / (1. + (s/t)^n)

# compute the gyncycle rhs and write it into f
function gyncycle_rhs!(y,p,f)
#function gyncycle_rhs!{T<:Real, S<:Real}(y::Vector{T}, p::Vector{S}, f::Vector{T})
#function gyncycle_rhs!{T<:Real}(y::Vector{T}, p::Vector{T}, f::Vector{T})
#function gyncycle_rhs!(y::Vector, p::Vector, f::Vector)
    #@assert length(p) == 114
    #@assert length(y) == length(f) == 33


    ### indices ###

    LH_pit    =  1
    LH_blood  =  2
    R_LH      =  3
    LH_R      =  4
    R_LH_des  =  5
    FSH_pit   =  6
    FSH_blood =  7
    R_FSH     =  8
    FSH_R     =  9
    R_FSH_des = 10
    s         = 11
    AF1       = 12
    AF2       = 13
    AF3       = 14
    AF4       = 15
    PrF       = 16
    OvF       = 17
    Sc1       = 18
    Sc2       = 19
    Lut1      = 20
    Lut2      = 21
    Lut3      = 22
    Lut4      = 23
    E2        = 24
    P4        = 25
    IhA       = 26
    IhB       = 27
    IhA_e     = 28
    G         = 29
    R_G_a     = 30
    R_G_i     = 31
    G_R_a     = 32
    G_R_i     = 33

    ### equations ###

    @inbounds begin

        # eq. 29a/b
        y_freq = p[93] * hminus(y[P4], p[94], p[95]) * (1. + p[96] * hplus(y[E2], p[97], p[98]))
        y_mass = p[99] * (hplus(y[E2], p[100], p[101]) + hminus(y[E2], p[102], p[103]))

        # eq. 1: LH in the pituitary (LH_pit)
        Syn_LH = (p[1] + p[2] * hplus(y[E2], p[3], p[4])) * hminus(y[P4], p[5], p[6])
        Rel_LH = (p[7] + p[8] * hplus(y[G_R_a], p[9], p[10]))
        f[LH_pit] = Syn_LH - Rel_LH * y[LH_pit]

        # eq. 2: LH in the blood (LH_blood) TODO: check occurence of y[LH_pit]
        f[LH_blood] = Rel_LH * y[LH_pit] / p[11] - (p[12] * y[R_LH] + p[13]) * y[LH_blood]

        # eq. 3: LH receptors (R_LH)
        f[R_LH] = p[14] * y[R_LH_des] - p[12] * y[LH_blood] * y[R_LH]

        # eq. 4: LH-receptor-complex (LH_R)
        f[LH_R] = p[12] * y[LH_blood] * y[R_LH] - p[15] * y[LH_R]

        # eq. 5: Internalized LH receptors (R_LH_des)
        f[R_LH_des] = p[15] * y[LH_R] - p[14] * y[R_LH_des]

        # eq. 6: FSH in the pituitary (FSH_pit)
        Syn_FSH = p[16] / (1. + (y[IhA_e]/p[17]) ^ p[18] + (y[IhB]/p[19]) ^ p[20]) * hminus(y_freq, p[21], p[22])
        Rel_FSH = (p[23] + p[24] * hplus(y[G_R_a], p[25], p[26]))
        f[FSH_pit] = Syn_FSH - Rel_FSH * y[FSH_pit]

        # eq. 7: FSH in the blood (FSH_blood)
        f[FSH_blood] = Rel_FSH * y[FSH_pit] / p[11] - (p[27] * y[R_FSH] + p[28]) * y[FSH_blood]

        # eq. 8: FSH receptors (R_FSH)
        f[R_FSH] = p[29] * y[R_FSH_des] - p[27] * y[FSH_blood] * y[R_FSH]

        # eq. 9: FSH-receptor-complex (FSH_R)
        f[FSH_R] = p[27] * y[FSH_blood] * y[R_FSH] - p[30] * y[FSH_R]

        # eq. 10: Internalized FSH receptors (R_FSH_des)
        f[R_FSH_des] = p[30] * y[FSH_R] - p[29] * y[R_FSH_des]

        # eq. 11: Follicular sensitivity to LH (s)
        f[s] = p[31] * hplus(y[FSH_blood], p[32], p[33]) - p[34] * hplus(y[P4], p[35], p[36]) * y[s]

        # eq. 12: Antral follicel develop. stage 1 (AF1)
        f[AF1] = p[37] * hplus(y[FSH_R], p[38], p[39]) - p[40] * y[FSH_R] * y[AF1]

        # eq. 13: Antral follicel develop. stage 2 (AF2)
        f[AF2] = p[40] * y[FSH_R] * y[AF1] - p[41] * (y[LH_R]/p[42]) ^ p[43] * y[s] * y[AF2]

        # eq. 14: Antral follicel develop. stage 3 (AF3)
        f[AF3] = p[41] * (y[LH_R]/p[42]) ^ p[43] * y[s] * y[AF2] +
                 p[44] * y[FSH_R] * y[AF3] * (1 - y[AF3]/p[45]) -
                 p[46] * (y[LH_R]/p[42]) ^ p[47] * y[s] * y[AF3]

        # eq. 15: Antral follicel develop. stage 4 (AF4)
        f[AF4] = p[46] * (y[LH_R]/p[42]) ^ p[47] * y[s] * y[AF3] +
                 p[48] * (y[LH_R]/p[42]) ^ p[49] * y[AF4] * (1. - y[AF4]/p[45]) -
                 p[50] * (y[LH_R]/p[42]) * y[s] * y[AF4]

        # eq. 16: Pre-ovulatory follicular stage (PrF)
        f[PrF] = p[50] * (y[LH_R]/p[42]) * y[s] * y[AF4] -
                 p[51] * (y[LH_R]/p[42]) ^ p[52] * y[s] * y[PrF]

        # eq. 17: Ovulatory follicular stage (OvF)
        f[OvF] = p[53] * (y[LH_R]/p[42]) ^ p[52] * y[s] * hplus(y[PrF], p[54], p[55]) - p[56] * y[OvF]

        # eq. 18: Ovulatory scar 1 (Sc1)
        f[Sc1] = p[57] * hplus(y[OvF], p[58], p[59]) - p[60] * y[Sc1]

        # eq. 19: Ovulatory scar 2 (Sc2)
        f[Sc2] = p[60] * y[Sc1] - p[61] * y[Sc2]

        # eq. 20: Development stage 1 of corpus luteum (Lut1)
        f[Lut1] = p[61] * y[Sc2] - p[62] * (1. + p[63] * hplus(y[G_R_a], p[64], p[65])) * y[Lut1]

        # eq. 21: Development stage 2 of corpus luteum (Lut2)
        f[Lut2] = p[62] * y[Lut1] - p[66] * (1. + p[63] * hplus(y[G_R_a], p[64], p[65])) * y[Lut2]

        # eq. 22: Development stage 3 of corpus luteum (Lut3)
        f[Lut3] = p[66] * y[Lut2] - p[67] * (1. + p[63] * hplus(y[G_R_a], p[64], p[65])) * y[Lut3]

        # eq. 23: Development stage 4 of corpus luteum (Lut4)
        f[Lut4] = p[67] * y[Lut3] - p[68] * (1. + p[63] * hplus(y[G_R_a], p[64], p[65])) * y[Lut4]

        # eq. 24: Estradiol blood level (E2)
        f[E2] = p[69] +
                p[70] * y[AF2] +
                p[71] * y[LH_blood] * y[AF3] +
                p[72] * y[AF4] +
                p[73] * y[LH_blood] * y[PrF] +
                p[74] * y[Lut1] +
                p[75] * y[Lut4] -
                p[76] * y[E2]

        # eq. 25: Progesterone blood level (P4)
        f[P4] = p[77] + p[78] * y[Lut4] - p[79] * y[P4]

        # eq. 26: Inhbin A blood level (IhA)
        f[IhA] = p[80] +
                 p[81] * y[PrF] +
                 p[82] * y[Sc1] +
                 p[83] * y[Lut1] +
                 p[84] * y[Lut2] +
                 p[85] * y[Lut3] +
                 p[86] * y[Lut4] -
                 p[87] * y[IhA]

        # eq. 27: Inhibin B blood level (IhB)
        f[IhB] = p[88] + p[89] * y[AF2] + p[90] * y[Sc2] - p[91] * y[IhB]

        # eq. 28: Effective inhibin A (IhA_e)
        f[IhA_e] = p[87] * y[IhA] - p[92] * y[IhA_e]

        # eq. 29: GnRH (G)
        f[G] = y_mass * y_freq - p[104] * y[G] * y[R_G_a] + p[105] * y[G_R_a] - p[106] * y[G]

        # eq. 30*: Active GnRH receptors (R_G_a)
        f[R_G_a] = p[105] * y[G_R_a] -
                   p[104] * y[G] * y[R_G_a] -
                   p[107] * y[R_G_a] +
                   p[108] * y[R_G_i]

        # eq. 31*: Inactive GnRH receptors (R_G_i)
        f[R_G_i] = p[109] +
                   p[107] * y[R_G_a] -
                   p[108] * y[R_G_i] +
                   p[110] * y[G_R_i] -
                   p[111] * y[R_G_i]

        # eq. 32: Active GnRH-receptor complex (G_R_a)
        f[G_R_a] = p[104] * y[R_G_a] * y[G] -
                   p[105] * y[G_R_a] -
                   p[112] * y[G_R_a] +
                   p[113] * y[G_R_i]

        # eq. 33: Inactive GnRH-receptor complex (G_R_i)
        f[G_R_i] = p[112] * y[G_R_a] -
                   p[113] * y[G_R_i] -
                   p[110] * y[G_R_i] -
                   p[114] * y[G_R_i]

    end

    return f
end

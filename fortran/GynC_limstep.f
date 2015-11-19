      subroutine  limstep(yall,y0,tspan,n,m,q) 

      IMPLICIT REAL*8 (A-H,O-Z)

      integer Iopt ( 30 ), IPos(n), Ifail(3)
      double precision Ropt ( 5 ), rtol(1), atol(1), Step, p(114)
      double precision yAll(n,m),y(n), ys(n), y0(n), tspan(m), q(114)

Cf2py intent(in) yall,y0,tspan,n,m,q
Cf2py intent(out) yall
Cf2py depend(n,m) yall
Cf2py depend(n) y0
Cf2py depend(m) tspan

      common /para/ p 

      EXTERNAL GynCycle_rhs, Jacobian
  
      do i=1,114 
      p(i)=q(i)
      end do
c-----------------------------------------------------------------------
c     Set some options for LIMEX    
c-----------------------------------------------------------------------
c     Integration monitor
      Iopt(1) = 0
c
c     Unit number for monitor output
      Iopt(2) = 0
c
c     Solution output
      Iopt(3) = 0
c
c     Unit number for solution output
      Iopt(4) = 0
c
c     Singular or nonsingular matrix B
      Iopt(5) = 1
c
c     Consistent initial value determination
      Iopt(6) = 0
c
c     Numerical or analytical computation of the Jacobian
      Iopt(7) = 0
c
c     Lower bandwith of the Jacobian if numerical evaluated
      Iopt(8) = n
c
c     Upper bandwith of the Jacobian if numerical evaluated
      Iopt(9) = n
c
c     Control of reuse of the Jacobian
      Iopt(10) = 0
c
c     Switch for error tolerances
      Iopt(11) = 0   
c
c     Switch for one step mode
      Iopt(12) = 0
c
c     Dense output options
      Iopt(13) = 0
      Iopt(14) = 0
      Iopt(15) = 0
c
c     Type of call
      Iopt(16) = 0
c
c     Switch for behavior of LIMEX on t_End
      Iopt(17) = 1
c
c     Generation of of a PostScript plot of the Jacobian   
      Iopt(18) = 0
c
c     Define maximal stepsize
      Ropt(1) = 1.0
       
      Ropt(2)=0
      
c damits schneller geht
      Ropt(3)= tspan(m)

      do i = 1, n
         IPos(i) = 0
      end do
c clear derivatives
      do i = 1, n
        ys(i) = 0.0d0
      end do

c     rtol, atol auf 10^-3
      rtol(1)=1.0e-3
      atol(1)=1.0e-3
      Stepsize=1.0e-6
c
      j = 1
      i=1
      t_begin=tspan(1)
      t_End=tspan(2)
c      y=y0
      do i=1,n
      yAll(i,1)=y0(i)
      end do

 
c       k=mexPrintf(line//achar(13)) 

      do j = 1, m-1
         t_begin=tspan(j)
         t_End=tspan(j+1)
         
c
         call LIMEX ( n, GynCycle_rhs, Jacobian, t_Begin, t_End, y0, ys,
     2                rTol, aTol, StepSize, Iopt, Ropt, IPos, IFail )

c      write(line,*) 't', t_begin, t_end, y0(1), y0(n)

c       write(line,*) 'hallo', 1.0
c       k=mexPrintf(line//achar(13)) 
c       return
         
         do i = 1, n
c          yAll(j*n+i)=y0(i)
          
          yAll(i,j)=y0(i)
         end do
     
   
      end do 


      
c
c      write ( *, '(6(/,a,i8))' )
c     2   ' F evaluations (without Jacobian) : ', Iopt(24),
c     3   ' F evaluations (Jacobian)         : ', Iopt(25),
c     4   ' LU decompositions                : ', Iopt(26),
c     5   ' back substitutions               : ', Iopt(27),
c     6   ' integration steps                : ', Iopt(28),
c     7   ' Jacobian matrix evaluations      : ', Iopt(29)


      return
      end 



      subroutine Jacobian ( n, t, y, ys, Jac, LDJac, ml, mu,
     2                      Full_or_Band, JacInfo )
c
      implicit double precision ( a-h, o-z )
c
c-----------------------------------------------------------------------
c
c     Defines the analytical Jacobian of the residual of the DAE
c
c     Within PAEON currently a dummy routine 
c
c-----------------------------------------------------------------------
c
      integer          n, LDJac, ml, mu, Full_or_Band, JacInfo
c
      double precision Jac
c
      dimension        Jac ( LDJAC, * )
c
      dimension        y ( * ), ys ( * )
c
c-----------------------------------------------------------------------
c
c     Dummy routine
c
c-----------------------------------------------------------------------
c
      JacInfo = - 1
c
c-----------------------------------------------------------------------
c
      return
      end


      subroutine GynCycle_rhs ( n, nz, t, y, f, b, ir, ic, Info )
c     y: initial values
c     f: rhs
c
c 
      implicit real*8 (a-h,o-z)
      dimension y(n),f(n), P(114), b(*) 
      integer ir(*), ic(*)
      common /para/ P


      i_LH_pit    =  1;
      i_LH_blood  =  2;
      i_R_LH      =  3;
      i_LH_R      =  4;
      i_R_LH_des  =  5;
      i_FSH_pit   =  6;
      i_FSH_blood =  7;
      i_R_FSH     =  8;
      i_FSH_R     =  9;
      i_R_FSH_des = 10;
      i_s         = 11;
      i_AF1       = 12;
      i_AF2       = 13;
      i_AF3       = 14;
      i_AF4       = 15;
      i_PrF       = 16;
      i_OvF       = 17;
      i_Sc1       = 18;
      i_Sc2       = 19;
      i_Lut1      = 20;
      i_Lut2      = 21;
      i_Lut3      = 22;
      i_Lut4      = 23;
      i_E2        = 24;
      i_P4        = 25;
      i_IhA       = 26;
      i_IhB       = 27;
      i_IhA_e     = 28;
      i_G         = 29;
      i_R_G_a     = 30;
      i_R_G_i     = 31;
      i_G_R_a     = 32; 
      i_G_R_i     = 33;

c---- GnRH frequency (freq)
 
      y_freq =  P(93) * Hminus ( y(i_P4), P(94), P(95) ) * 
     2 ( 1.0d0 + P(96) * Hplus ( y(i_E2), P(97), P(98) ) ) 

c---- GnRH mass (mass)

      y_mass =   P(99) * (   Hplus  ( y(i_E2), P(100), P(101) ) 
     2        + Hminus ( y(i_E2), P(102), P(103) ) ) 

c---- Equation  1 : LH in the pituitary (LH_pit)

      Syn_LH =   ( P(1) + P(2) * Hplus ( y(i_E2), P(3), P(4) ) )
     2         * Hminus ( y(i_P4), P(5), P(6) )

      Rel_LH =  ( P(7) + P(8) * Hplus ( y(i_G_R_a), P(9), P(10) ) )
  
      f(i_LH_pit) = Syn_LH - Rel_LH * y(i_LH_pit)

c---- Equation  2 : LH in the blood (LH_blood) 

      f(i_LH_blood) =   Rel_LH * y(i_LH_pit) / P(11)
     2    - ( P(12) * y(i_R_LH) + P(13) ) * y(i_LH_blood)

c---- Equation  3 : LH receptors (R_LH) 

      f(i_R_LH) =   P(14) * y(i_R_LH_des) 
     2                             - P(12) * y(i_LH_blood) * y(i_R_LH)

c---- Equation  4 : LH-receptor-complex (LH_R)

      f(i_LH_R) =  P(12) * y(i_LH_blood) * y(i_R_LH) - P(15) * y(i_LH_R)

c---- Equation  5 : Internalized LH receptors (R_LH_des)

      f(i_R_LH_des) = P(15) * y(i_LH_R) - P(14) * y(i_R_LH_des)

c---- Equation  6 : FSH in the pituitary (FSH_pit)

      Syn_FSH =   P(16) / ( 1.0d0 + ( y(i_IhA_e) / P(17) ) ** P(18) 
     2                      + ( y(i_IhB)   / P(19) ) ** P(20) ) 
     3      * Hminus ( y_freq, P(21), P(22) )

      Rel_FSH = ( P(23) + P(24) * Hplus ( y(i_G_R_a), P(25), P(26) ) )

      f(i_FSH_pit) = Syn_FSH - Rel_FSH * y(i_FSH_pit)
c       write(*,*) ' hh ',y(i_E2), P(97), P(98) 
c       write(*,*) ' aa', Hplus ( y(i_E2), P(97), P(98) )
c---- Equation  7 : FSH in the blood (FSH_blood)

      f(i_FSH_blood) =  Rel_FSH * y(i_FSH_pit) / P(11) 
     2   - ( P(27) * y(i_R_FSH) + P(28) ) * y(i_FSH_blood)

c---- Equation  8 : FSH receptors (R_FSH)

      f(i_R_FSH) =   P(29) * y(i_R_FSH_des)  - 
     2         P(27) * y(i_FSH_blood) * y(i_R_FSH)

c---- Equation  9 : FSH-receptor-complex (FSH_R)

      f(i_FSH_R) = P(27) * y(i_FSH_blood) * y(i_R_FSH) - 
     2                                     P(30) * y(i_FSH_R)

c---- Equation 10 : Internalized FSH receptors (R_FSH_des)

      f(i_R_FSH_des) = P(30) * y(i_FSH_R) - P(29) * y(i_R_FSH_des)

c---- Equation 11 : Follicular sensitivity to LH (s)

      f(i_s) =   P(31) * Hplus ( y(i_FSH_blood), P(32), P(33) ) 
     2        - P(34) * Hplus ( y(i_P4), P(35), P(36) ) * y(i_s)

c---- Equation 12 : Antral follicel develop. stage 1 (AF1)

      f(i_AF1) =   P(37) * Hplus ( y(i_FSH_R), P(38), P(39) ) 
     2        - P(40) * y(i_FSH_R) * y(i_AF1)

c---- Equation 13 : Antral follicel develop. stage 2 (AF2)

      f(i_AF2) =   P(40) * y(i_FSH_R) * y(i_AF1)
     2     - P(41) * ( y(i_LH_R) / P(42) ) ** P(43) * y(i_s) * y(i_AF2)

c---- Equation 14 : Antral follicel develop. stage 3 (AF3)

      f(i_AF3) = P(41) * ( y(i_LH_R) / P(42) ) ** P(43) *y(i_s)*y(i_AF2)
     2      + P(44) * y(i_FSH_R) * y(i_AF3) * ( 1 - y(i_AF3) / P(45) )
     2      - P(46) * ( y(i_LH_R) / P(42) ) ** P(47) * y(i_s) * y(i_AF3)

c---- Equation 15 : Antral follicel develop. stage 4 (AF4)

      f(i_AF4) = P(46) * ( y(i_LH_R) / P(42) ) ** P(47) *y(i_s)*y(i_AF3)
     2   + P(48) * ( y(i_LH_R) / P(42) ) ** P(49) * y(i_AF4) 
     3                               * ( 1.0d0 - y(i_AF4) / P(45) ) 
     4   - P(50) * ( y(i_LH_R) / P(42) ) * y(i_s) * y(i_AF4)

c---- Equation 16 : Pre-ovulatory follicular stage (PrF)

      f(i_PrF) =   P(50) * ( y(i_LH_R) / P(42) ) * y(i_s) * y(i_AF4)
     2           - P(51) * ( y(i_LH_R) / P(42) ) ** P(52) 
     3             * y(i_s) * y(i_PrF)

c---- Equation 17 : Ovulatory follicular stage (OvF)

      f(i_OvF) =   P(53) * ( y(i_LH_R) / P(42) ) ** P(52) * y(i_s)
     2            * Hplus ( y(i_PrF), P(54), P(55) )
     3           - P(56) * y(i_OvF)

c---- Equation 18 : Ovulatory scar 1 (Sc1)

      f(i_Sc1) =  P(57) * Hplus ( y(i_OvF), P(58), P(59) )
     2                                  - P(60) * y(i_Sc1)

c---- Equation 19 : Ovulatory scar 2 (Sc2)

      f(i_Sc2) = P(60) * y(i_Sc1) - P(61) * y(i_Sc2)

c---- Equation 20 : Development stage 1 of corpus luteum (Lut1)

      f(i_Lut1) =P(61) * y(i_Sc2) 
     2   - P(62) * ( 1.0d0 + P(63) * 
     3   Hplus ( y(i_G_R_a), P(64), P(65) ) ) * y(i_Lut1)

c---- Equation 21 : Development stage 2 of corpus luteum (Lut2)
    
      f(i_Lut2) =  P(62) * y(i_Lut1) - P(66) * ( 1.0d0 + P(63) 
     2     * Hplus ( y(i_G_R_a), P(64), P(65) ) ) * y(i_Lut2)

c---- Equation 22 : Development stage 3 of corpus luteum (Lut3)
    
      f(i_Lut3) = P(66) * y(i_Lut2) - P(67) * ( 1.0d0 + P(63) * 
     2     Hplus ( y(i_G_R_a), P(64), P(65) ) ) * y(i_Lut3)

c---- Equation 23 : Development stage 4 of corpus luteum (Lut4)
   
      f(i_Lut4) = P(67) * y(i_Lut3) - P(68) * ( 1.0d0 + P(63) 
     2     * Hplus ( y(i_G_R_a), P(64), P(65) ) ) * y(i_Lut4)

c---- Equation 24 : Estradiol blood level (E2)

      f(i_E2) =   P(69)  + P(70) * y(i_AF2) 
     2           + P(71) * y(i_LH_blood) * y(i_AF3) 
     3           + P(72) * y(i_AF4) 
     4           + P(73) * y(i_LH_blood) * y(i_PrF) 
     5           + P(74) * y(i_Lut1) 
     6           + P(75) * y(i_Lut4) 
     7           - P(76) * y(i_E2)

c---- Equation 25 : Progesterone blood level (P4)

      f(i_P4) = P(77) + P(78) * y(i_Lut4) - P(79) * y(i_P4)

c---- Equation 26 : Inhbin A blood level (IhA)

      f(i_IhA) =   P(80) 
     2           + P(81) * y(i_PrF) 
     3           + P(82) * y(i_Sc1) 
     4           + P(83) * y(i_Lut1) 
     5           + P(84) * y(i_Lut2) 
     6           + P(85) * y(i_Lut3) 
     7           + P(86) * y(i_Lut4)  
     8           - P(87) * y(i_IhA)

c---- Equation 27 : Inhibin B blood level (IhB)

      f(i_IhB) =   P(88)  + P(89) * y(i_AF2)  
     2         + P(90) * y(i_Sc2) 
     3          - P(91) * y(i_IhB)

c---- Equation 28 : Effective inhibin A (IhA_e)

      f(i_IhA_e) = P(87) * y(i_IhA) - P(92) * y(i_IhA_e)

c---- Equation 31 : GnRH (G)	  

      f(i_G) =   y_mass * y_freq 
     2         - P(104) * y(i_G) * y(i_R_G_a) 
     3        + P(105) * y(i_G_R_a) 
     4         - P(106) * y(i_G)

c---- Equation 32 : Active GnRH receptors (R_G_a)

      f(i_R_G_a) =   P(105) * y(i_G_R_a) 
     2             - P(104) * y(i_G) * y(i_R_G_a)  
     3             - P(107) * y(i_R_G_a) 
     4             + P(108) * y(i_R_G_i)

c---- Equation 33 : Inactive GnRH receptors (R_G_i)	 

      f(i_R_G_i) =   P(109) 
     2      + P(107) * y(i_R_G_a)
     3      - P(108) * y(i_R_G_i)
     4      + P(110) * y(i_G_R_i) 
     5     - P(111) * y(i_R_G_i)

c---- Equation 34 : Active GnRH-receptor complex (G_R_a)

      f(i_G_R_a) =   P(104) * y(i_G) * y(i_R_G_a)
     2             - P(105) * y(i_G_R_a)
     3             - P(112) * y(i_G_R_a)
     4             + P(113) * y(i_G_R_i)

c---- Equation 35 : Inactive GnRH-receptor complex (G_R_i)

      f(i_G_R_i) =   P(112) * y(i_G_R_a) 
     2             - P(113) * y(i_G_R_i)
     3             - P(110) * y(i_G_R_i)
     4             - P(114) * y(i_G_R_i)

c      write(*,*) 'f6', f(6), y(6)
  
      nz=n
      do i = 1, n
         ir(i) = i
         ic(i) = i
         b(i)  = 1.0d0
      end do
      Info=0
      return 
      end 

      double precision function Hplus ( S, T, n )
      implicit none
      double precision  S, T, n
      Hplus = ( S / T ) ** n / ( 1.0d0 + ( S / T ) ** n )
      return
      end

      double precision function Hminus ( S, T, n )
      implicit none
      double precision  S, T, n
      Hminus = 1.0d0 / ( 1.0d0 + ( S / T ) ** n )
      return
      end


      



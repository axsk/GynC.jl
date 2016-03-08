standard:
	#f2py -c  -m GynC_limstep  --opt="-O3" --fcompiler=gnu95 GynC_limstep.f LIMEX4_3A.f LIMEX4_3_Dense.f LIMEX4_3_Auxiliaries.f decsol.f
	gfortran  GynC_limstep.f LIMEX4_3A.f LIMEX4_3_Dense.f LIMEX4_3_Auxiliaries.f decsol.f -shared -fPIC -o GynC.so 

period:
	#f2py -c -m GynC_limstepPeriod GynC_limstepPeriod.f LIMEX4_3A.f LIMEX4_3_Dense.f LIMEX4_3_Auxiliaries.f Period.f decsol.f


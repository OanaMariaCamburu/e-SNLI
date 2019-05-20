import scipy.stats
import numpy as np


# with my model
Breaking_b =[55.67, 60.84, 50.31, 48.65, 53.23] 
Breaking_me = [62.9, 56.29, 55.28, 58.5, 51.64]
mr_ttest = scipy.stats.ttest_ind(Breaking_b, Breaking_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(Breaking_me) - np.mean(Breaking_b)
print("Breaking ", mr_ttest)
# ('Breaking ', Ttest_indResult(statistic=-1.1205000571143193, pvalue=0.2956371604272169))

'''
# with my model
MR_b =[78.4, 78.2, 77.8, 78.4, 78.1] 
MR_me = [78.4, 77.7, 77.6, 77.9, 77.2]
mr_ttest = scipy.stats.ttest_ind(MR_b, MR_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MR_me) - np.mean(MR_b)
print("MR ", mr_ttest)


CR_b = [81.5, 81.3, 81.3, 81.1, 81.2]
CR_me =[81.1, 81.4, 81.3, 81.2, 81.5] 
cr_ttest = scipy.stats.ttest_ind(CR_b, CR_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(CR_me) - np.mean(CR_b)
print("CR ", cr_ttest)


SUBJ_b = [92.6, 92.5, 92.6, 92.3, 92.3]
SUBJ_me = [92.1, 92, 92, 92.1, 92.5]
subj_ttest = scipy.stats.ttest_ind(SUBJ_b, SUBJ_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SUBJ_me) - np.mean(SUBJ_b)
print("SUBJ ", subj_ttest)


MPQA_b = [88.3, 88.5, 88.7, 88.2, 88.6]
MPQA_me = [88.9, 88.9, 89, 88.6, 88.5]
MPQA_ttest = scipy.stats.ttest_ind(MPQA_b, MPQA_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MPQA_me) - np.mean(MPQA_b)
print("MPQA ", MPQA_ttest)


SST2_b = [82.2, 82.3, 81.8, 82, 82.3] 
SST2_me = [81.8, 82.3, 81.5, 82.2, 81.4]
SST2_ttest = scipy.stats.ttest_ind(SST2_b, SST2_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SST2_me) - np.mean(SST2_b)
print("SST2_ttest ", SST2_ttest)


TREC_b = [88.6, 89.4, 89.2, 90, 89.4]
TREC_me = [90, 90.8, 90, 89.8, 89.4]
TREC_ttest = scipy.stats.ttest_ind(TREC_b, TREC_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(TREC_me) - np.mean(TREC_b)
print("TREC_ttest ", TREC_ttest)



MRPC_acc_b = [75, 75.3, 73.7, 75.3, 74.84]
MRPC_acc_me = [74.8, 75.6, 75.6, 75.3, 76.5]
MRPC_acc_ttest = scipy.stats.ttest_ind(MRPC_acc_b, MRPC_acc_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MRPC_acc_me) - np.mean(MRPC_acc_b)
print("MRPC_acc_ttest ", MRPC_acc_ttest)


MRPC_F1_b = [82.9, 82.7, 82.4, 83.1, 82.6]	
MRPC_F1_me = [82.9, 83.3, 83.4, 83.1, 83.5]
MRPC_F1_ttest = scipy.stats.ttest_ind(MRPC_F1_b, MRPC_F1_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MRPC_F1_me) - np.mean(MRPC_F1_b)
print("MRPC_F1_ttest ", MRPC_F1_ttest)


SICKEntailment_b = [85.7, 86.4, 85.8, 85.7, 86.2]
SICKEntailment_me = [85.4, 85.8, 85.8, 85.8, 86.8]
SICKEntailment_ttest = scipy.stats.ttest_ind(SICKEntailment_b, SICKEntailment_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SICKEntailment_me) - np.mean(SICKEntailment_b)
print("SICKEntailment_ttest ", SICKEntailment_ttest)


SICK_R_pearson_b = [0.89, 0.885, 0.886, 0.886, 0.888]
SICK_R_pearson_me = [0.891, 0.892, 0.894, 0.893, 0.893]
SICK_R_pearson_ttest = scipy.stats.ttest_ind(SICK_R_pearson_b, SICK_R_pearson_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SICK_R_pearson_me) - np.mean(SICK_R_pearson_b)
print("SICK_R_pearson_ttest ", SICK_R_pearson_ttest)


STS14_pearson_b = [0.65, 0.65, 0.65, 0.65, 0.66]
STS14_pearson_me = [0.68, 0.67, 0.68, 0.68, 0.67]
STS14_pearson_ttest = scipy.stats.ttest_ind(STS14_pearson_b, STS14_pearson_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(STS14_pearson_me) - np.mean(STS14_pearson_b)
print("STS14_pearson_ttest ", STS14_pearson_ttest)


STS14_spearman_b = [0.63, 0.63, 0.63, 0.63, 0.64]
STS14_spearman_me = [0.65, 0.64, 0.65, 0.65, 0.64]
STS14_spearman_ttest = scipy.stats.ttest_ind(STS14_spearman_b, STS14_spearman_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(STS14_spearman_me) - np.mean(STS14_spearman_b)
print("STS14_spearman_ttest ", STS14_spearman_ttest)

'''

'''  <0.05
diff  -0.42
('MR ', Ttest_indResult(statistic=1.8597912596342565, pvalue=0.10970481733423301))

diff  0.02
('CR ', Ttest_indResult(statistic=-0.20628424925171535, pvalue=0.8417425653416252))

diff  -0.32
('SUBJ ', Ttest_indResult(statistic=2.785242495291109, pvalue=0.025876578712689995)) !!

diff  0.32
('MPQA ', Ttest_indResult(statistic=-2.385139175999818, pvalue=0.04424907297982854)) !!

diff  -0.28
('SST2_ttest ', Ttest_indResult(statistic=1.3662601021279561, pvalue=0.2198563514987347))

diff  0.68
('TREC_ttest ', Ttest_indResult(statistic=-2.1249999999999774, pvalue=0.0663154365723148))

diff  0.732
('MRPC_acc_ttest ', Ttest_indResult(statistic=-1.8076333412556422, pvalue=0.10844345460355041))

diff  0.5
('MRPC_F1_ttest ', Ttest_indResult(statistic=-3.089010316076006, pvalue=0.015150627240608113)) !!

diff  -0.04
('SICKEntailment_ttest ', Ttest_indResult(statistic=0.14605934866806736, pvalue=0.8882031667608185))

diff  0.0056
('SICK_R_pearson_ttest ', Ttest_indResult(statistic=-5.439200829200614, pvalue=0.0013328314936747142)) !!

diff  0.024
('STS14_pearson_ttest ', Ttest_indResult(statistic=-7.589466384404111, pvalue=7.832497380416784e-05))

diff  0.014
('STS14_spearman_ttest ', Ttest_indResult(statistic=-4.427188724235731, pvalue=0.0024270116218849964)) !!
'''

'''
# with infersent_baseline vs autoenc baseline as _me
MR_b =[78.4, 78.2, 77.8, 78.4, 78.1] 
MR_me = [76.2, 75.9, 75.7, 76, 75.9]
mr_ttest = scipy.stats.ttest_ind(MR_b, MR_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MR_me) - np.mean(MR_b)
print("MR ", mr_ttest)


CR_b = [81.5, 81.3, 81.3, 81.1, 81.2]
CR_me =[78.7, 79.7, 79.3, 79.3, 79.3] 
cr_ttest = scipy.stats.ttest_ind(CR_b, CR_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(CR_me) - np.mean(CR_b)
print("CR ", cr_ttest)


SUBJ_b = [92.6, 92.5, 92.6, 92.3, 92.3]
SUBJ_me = [91.8, 91.8, 92.1, 91.4, 91.5]
subj_ttest = scipy.stats.ttest_ind(SUBJ_b, SUBJ_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SUBJ_me) - np.mean(SUBJ_b)
print("SUBJ ", subj_ttest)


MPQA_b = [88.3, 88.5, 88.7, 88.2, 88.6]
MPQA_me = [88.2, 87.9, 88.3, 88.5, 87.9]
MPQA_ttest = scipy.stats.ttest_ind(MPQA_b, MPQA_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MPQA_me) - np.mean(MPQA_b)
print("MPQA ", MPQA_ttest)


SST2_b = [82.2, 82.3, 81.8, 82, 82.3] 
SST2_me = [81.1, 80.3, 80.7, 81.6, 80.8]
SST2_ttest = scipy.stats.ttest_ind(SST2_b, SST2_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SST2_me) - np.mean(SST2_b)
print("SST2_ttest ", SST2_ttest)


TREC_b = [88.6, 89.4, 89.2, 90, 89.4]
TREC_me = [90, 91.4, 90.4, 90.4, 90.4]
TREC_ttest = scipy.stats.ttest_ind(TREC_b, TREC_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(TREC_me) - np.mean(TREC_b)
print("TREC_ttest ", TREC_ttest)



MRPC_acc_b = [75, 75.3, 73.7, 75.3, 74.84]
MRPC_acc_me = [75.2, 76.7, 77.5, 75.5, 76.1]
MRPC_acc_ttest = scipy.stats.ttest_ind(MRPC_acc_b, MRPC_acc_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MRPC_acc_me) - np.mean(MRPC_acc_b)
print("MRPC_acc_ttest ", MRPC_acc_ttest)


MRPC_F1_b = [82.9, 82.7, 82.4, 83.1, 82.6]	
MRPC_F1_me = [81.2, 83, 84.1, 81.3, 82.8]
MRPC_F1_ttest = scipy.stats.ttest_ind(MRPC_F1_b, MRPC_F1_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MRPC_F1_me) - np.mean(MRPC_F1_b)
print("MRPC_F1_ttest ", MRPC_F1_ttest)


SICKEntailment_b = [85.7, 86.4, 85.8, 85.7, 86.2]
SICKEntailment_me = [85.8, 85.7, 85, 85.7, 85.7]
SICKEntailment_ttest = scipy.stats.ttest_ind(SICKEntailment_b, SICKEntailment_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SICKEntailment_me) - np.mean(SICKEntailment_b)
print("SICKEntailment_ttest ", SICKEntailment_ttest)


SICK_R_pearson_b = [0.89, 0.885, 0.886, 0.886, 0.888]
SICK_R_pearson_me = [0.884, 0.88, 0.878, 0.878, 0.882]
SICK_R_pearson_ttest = scipy.stats.ttest_ind(SICK_R_pearson_b, SICK_R_pearson_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SICK_R_pearson_me) - np.mean(SICK_R_pearson_b)
print("SICK_R_pearson_ttest ", SICK_R_pearson_ttest)


STS14_pearson_b = [0.65, 0.65, 0.65, 0.65, 0.66]
STS14_pearson_me = [0.47, 0.48, 0.52, 0.52, 0.49]
STS14_pearson_ttest = scipy.stats.ttest_ind(STS14_pearson_b, STS14_pearson_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(STS14_pearson_me) - np.mean(STS14_pearson_b)
print("STS14_pearson_ttest ", STS14_pearson_ttest)


STS14_spearman_b = [0.63, 0.63, 0.63, 0.63, 0.64]
STS14_spearman_me = [0.48, 0.49, 0.52, 0.51, 0.5]
STS14_spearman_ttest = scipy.stats.ttest_ind(STS14_spearman_b, STS14_spearman_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(STS14_spearman_me) - np.mean(STS14_spearman_b)
print("STS14_spearman_ttest ", STS14_spearman_ttest)
'''


'''
diff  -2.24
('MR ', Ttest_indResult(statistic=16.25066800246417, pvalue=5.221402450370503e-07)) !!

diff  -2.02
('CR ', Ttest_indResult(statistic=11.662475437630418, pvalue=5.341062563029558e-05)) !!

diff  -0.74
('SUBJ ', Ttest_indResult(statistic=5.232590180780583, pvalue=0.001770553275536762)) !!

diff  -0.3
('MPQA ', Ttest_indResult(statistic=2.013468165641975, pvalue=0.08065680512392243))

diff  -1.22
('SST2_ttest ', Ttest_indResult(statistic=5.1371267185058365, pvalue=0.0027152007700303833)) !!

diff  1.2
('TREC_ttest ', Ttest_indResult(statistic=-3.7068123792912884, pvalue=0.005999092424898797)) !!

diff  1.372
('MRPC_acc_ttest ', Ttest_indResult(statistic=-2.6939065096322263, pvalue=0.029985501243120778)) !!

diff  -0.26
('MRPC_F1_ttest ', Ttest_indResult(statistic=0.46251924352716833, pvalue=0.6657309918819334))

diff  -0.38
('SICKEntailment_ttest ', Ttest_indResult(statistic=1.8542101386022545, pvalue=0.10083643929310204))

diff  -0.0066
('SICK_R_pearson_ttest ', Ttest_indResult(statistic=4.490731195102523, pvalue=0.002385085505598508)) !!

diff  -0.156
('STS14_pearson_ttest ', Ttest_indResult(statistic=14.874016392231228, pvalue=7.320708999128239e-05)) !!

diff  -0.132
('STS14_spearman_ttest ', Ttest_indResult(statistic=17.962924780409956, pvalue=1.8293734491721762e-05)) !!
'''


'''
# with baseline autoencoder my model
MR_b =[78.4, 78.2, 77.8, 78.4, 78.1] 
MR_me = [78.4, 77.7, 77.6, 77.9, 77.2]
mr_ttest = scipy.stats.ttest_ind(MR_b, MR_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MR_me) - np.mean(MR_b)
print("MR ", mr_ttest)


CR_b = [81.5, 81.3, 81.3, 81.1, 81.2]
CR_me =[81.1, 81.4, 81.3, 81.2, 81.5] 
cr_ttest = scipy.stats.ttest_ind(CR_b, CR_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(CR_me) - np.mean(CR_b)
print("CR ", cr_ttest)


SUBJ_b = [92.6, 92.5, 92.6, 92.3, 92.3]
SUBJ_me = [92.1, 92, 92, 92.1, 92.5]
subj_ttest = scipy.stats.ttest_ind(SUBJ_b, SUBJ_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SUBJ_me) - np.mean(SUBJ_b)
print("SUBJ ", subj_ttest)


MPQA_b = [88.3, 88.5, 88.7, 88.2, 88.6]
MPQA_me = [88.9, 88.9, 89, 88.6, 88.5]
MPQA_ttest = scipy.stats.ttest_ind(MPQA_b, MPQA_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MPQA_me) - np.mean(MPQA_b)
print("MPQA ", MPQA_ttest)


SST2_b = [82.2, 82.3, 81.8, 82, 82.3] 
SST2_me = [81.8, 82.3, 81.5, 82.2, 81.4]
SST2_ttest = scipy.stats.ttest_ind(SST2_b, SST2_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SST2_me) - np.mean(SST2_b)
print("SST2_ttest ", SST2_ttest)


TREC_b = [88.6, 89.4, 89.2, 90, 89.4]
TREC_me = [90, 90.8, 90, 89.8, 89.4]
TREC_ttest = scipy.stats.ttest_ind(TREC_b, TREC_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(TREC_me) - np.mean(TREC_b)
print("TREC_ttest ", TREC_ttest)



MRPC_acc_b = [75, 75.3, 73.7, 75.3, 74.84]
MRPC_acc_me = [74.8, 75.6, 75.6, 75.3, 76.5]
MRPC_acc_ttest = scipy.stats.ttest_ind(MRPC_acc_b, MRPC_acc_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MRPC_acc_me) - np.mean(MRPC_acc_b)
print("MRPC_acc_ttest ", MRPC_acc_ttest)


MRPC_F1_b = [82.9, 82.7, 82.4, 83.1, 82.6]	
MRPC_F1_me = [82.9, 83.3, 83.4, 83.1, 83.5]
MRPC_F1_ttest = scipy.stats.ttest_ind(MRPC_F1_b, MRPC_F1_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(MRPC_F1_me) - np.mean(MRPC_F1_b)
print("MRPC_F1_ttest ", MRPC_F1_ttest)


SICKEntailment_b = [85.7, 86.4, 85.8, 85.7, 86.2]
SICKEntailment_me = [85.4, 85.8, 85.8, 85.8, 86.8]
SICKEntailment_ttest = scipy.stats.ttest_ind(SICKEntailment_b, SICKEntailment_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SICKEntailment_me) - np.mean(SICKEntailment_b)
print("SICKEntailment_ttest ", SICKEntailment_ttest)


SICK_R_pearson_b = [0.89, 0.885, 0.886, 0.886, 0.888]
SICK_R_pearson_me = [0.891, 0.892, 0.894, 0.893, 0.893]
SICK_R_pearson_ttest = scipy.stats.ttest_ind(SICK_R_pearson_b, SICK_R_pearson_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(SICK_R_pearson_me) - np.mean(SICK_R_pearson_b)
print("SICK_R_pearson_ttest ", SICK_R_pearson_ttest)


STS14_pearson_b = [0.65, 0.65, 0.65, 0.65, 0.66]
STS14_pearson_me = [0.68, 0.67, 0.68, 0.68, 0.67]
STS14_pearson_ttest = scipy.stats.ttest_ind(STS14_pearson_b, STS14_pearson_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(STS14_pearson_me) - np.mean(STS14_pearson_b)
print("STS14_pearson_ttest ", STS14_pearson_ttest)


STS14_spearman_b = [0.63, 0.63, 0.63, 0.63, 0.64]
STS14_spearman_me = [0.65, 0.64, 0.65, 0.65, 0.64]
STS14_spearman_ttest = scipy.stats.ttest_ind(STS14_spearman_b, STS14_spearman_me, axis=0, equal_var=False)
print "\ndiff ", np.mean(STS14_spearman_me) - np.mean(STS14_spearman_b)
print("STS14_spearman_ttest ", STS14_spearman_ttest)
'''

'''
# breaking NLI
breaking_b = [55.67, 60.84, 50.31, 48.65, 53.23]
breaking_me = [62.9, 56.29, 55.28, 58.5, 51.64]
breaking_ttest = scipy.stats.ttest_ind(breaking_b, breaking_me, axis=0, equal_var=False)
print("breaking_ttest ", breaking_ttest)
# ('breaking_ttest ', Ttest_indResult(statistic=-1.1205000571143193, pvalue=0.2956371604272169))
'''
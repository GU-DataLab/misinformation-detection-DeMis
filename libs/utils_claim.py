#!/usr/bin/env python
# coding: utf-8

'''
@title: Utilities for working on claim data
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@description: Utilities for working on claim data gathered from Google Fact Checker API
'''


# Already minus 1 of the id
# TARGET_CLAIM_IDS = {
#     "home-remedies": [
#         601, 583, 522, 521, 506, 502,
#         360, 336, 283, 259, 145, 243
#         # 145, 243, 259, 283, 336, 360, 506, 583
#         ],
#     "weather": [135, 545, 586],
#     "covidlies-top4": [2, 24, 9, 14],
#     "spread": [
#         70, 93, 117, 345, 633
#     ]
# }


def is_same_claim(claim_A, claim_B):
    if set(claim_A.keys()) != set(claim_B.keys()):
        return False

    for k in claim_A.keys():
        if claim_A[k] != claim_B[k]:
            return False

    return True

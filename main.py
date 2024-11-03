from bisect import bisect_right


class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        return bisect_right(range(x+1),0,key = lambda q:q*q-x)-1


    def mySqrt_2(self,x):
        if x == 0:
            return 0
        if x<4:
            return 1
        left = 0
        right = x
        while(left<right):
            mid = (left+right)//2
            mult = mid*mid
            mult_plus = (mid+1)*(mid+1)
            if mult == x:
                return mid
            if mult<x:
                if mult_plus>x:
                    return mid
                left= mid+1
            if mult>x:
                right = mid-1
        return right


sol = Solution()
print(sol.mySqrt_2(16))
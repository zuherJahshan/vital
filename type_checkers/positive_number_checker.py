class PositiveNumberChecker():
    def __call__(self, number):
        if number <= 0:
            raise ValueError("Number must be positive")
        return number
    
class FloatBetweenZeroAndOneChecker():
    def __call__(self, number):
        if number <= 0 or number >= 1:
            raise ValueError("Number must be in the range [0, 1]")
        return number
    
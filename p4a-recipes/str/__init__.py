from pythonforandroid.recipe import Recipe

class SrtRecipe(Recipe):
    version = '0.0.1'
    url = None
    depends = []

    def should_build(self, arch):
        return False

    def build_arch(self, arch):
        pass

recipe = SrtRecipe()
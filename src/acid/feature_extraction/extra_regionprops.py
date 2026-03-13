from .measure_object_instensity import median_intensity, skew_intensity, lower_quartile_intensity, upper_quartile_intensity, mad_intensity, integrated_intensity

def regionpros_extra_props():
    props = [median_intensity, skew_intensity, lower_quartile_intensity, upper_quartile_intensity,
             mad_intensity, integrated_intensity]
    return props


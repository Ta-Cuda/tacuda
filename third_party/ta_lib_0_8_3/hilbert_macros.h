/* Provides common mathematical or analysis functions.
 *
 * These functions are all PRIVATE to ta-lib and should
 * never be called directly by the library users.
 */
#define HILBERT_VARIABLES(varName) \
   ARRAY_LOCAL(varName##_Odd,3); \
   ARRAY_LOCAL(varName##_Even, 3); \
   double varName; \
   double prev_##varName##_Odd; \
   double prev_##varName##_Even; \
   double prev_##varName##_input_Odd; \
   double prev_##varName##_input_Even

#define INIT_HILBERT_VARIABLES(varName) { \
      varName##_Odd [0] = 0.0; \
      varName##_Odd [1] = 0.0; \
      varName##_Odd [2] = 0.0; \
      varName##_Even[0] = 0.0; \
      varName##_Even[1] = 0.0; \
      varName##_Even[2] = 0.0; \
      varName = 0.0; \
      prev_##varName##_Odd        = 0.0; \
      prev_##varName##_Even       = 0.0; \
      prev_##varName##_input_Odd  = 0.0; \
      prev_##varName##_input_Even = 0.0; \
      }

#define DO_HILBERT_TRANSFORM(varName,input,OddOrEvenId) {\
         hilbertTempReal = a * input; \
         varName = -varName##_##OddOrEvenId[hilbertIdx]; \
         varName##_##OddOrEvenId[hilbertIdx] = hilbertTempReal; \
         varName += hilbertTempReal; \
         varName -= prev_##varName##_##OddOrEvenId; \
         prev_##varName##_##OddOrEvenId = b * prev_##varName##_input_##OddOrEvenId; \
         varName += prev_##varName##_##OddOrEvenId; \
         prev_##varName##_input_##OddOrEvenId = input; \
         varName *= adjustedPrevPeriod; \
         }

#define DO_HILBERT_ODD(varName,input)  DO_HILBERT_TRANSFORM(varName,input,Odd)
#define DO_HILBERT_EVEN(varName,input) DO_HILBERT_TRANSFORM(varName,input,Even)

/* Convert a period into the equivalent k:
 *
 *    k = 2 / (period + 1)
 *
 * Useful to calculate the smoothing factor 'k' of an EMA.
 */

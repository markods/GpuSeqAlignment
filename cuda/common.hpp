// missing Common.cpp file on purpose, since whole program optimization is disabled
#pragma once
#include <cstdio>
#include <chrono>
#include <memory>
#include <limits>
#include <unordered_map>


// number of streaming multiprocessors (sm-s) and cores per sm
constexpr int MPROCS = 28;
constexpr int CORES = 128;
// number of threads in warp
constexpr int WARPSZ = 32;

// get the specified element from the given linearized matrix
#define el( mat, cols, i, j ) ( mat[(i)*(cols) + (j)] )

// for diagnostic purposes
inline void PrintMatrix(
   const int* const matrix,
   const int rows,
   const int cols
)
{
   printf( "\n" );
   for( int i = 0; i < rows; i++ )
   {
      for( int j = 0; j < cols; j++ )
      {
         printf( "%3d ", el(matrix,cols, i,j) );
      }
      printf( "\n" );
   }
   fflush(stdout);
}

// for diagnostic purposes
inline void ZeroOutMatrix(
   int* const matrix,
   const int rows,
   const int cols
) noexcept
{
   for( int i = 0; i < rows; i++ )
   for( int j = 0; j < cols; j++ )
   {
      el(matrix,cols, i,j) = 0;
   }
}


// TODO: remove
// block substitution matrix
#define SUBSTSZ 24

// TODO: test performance of min2, max2 and max3 without branching
// +   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

// calculate the minimum of two numbers
inline const int& min2( const int& a, const int& b ) noexcept
{
   return ( a < b ) ? a : b;
}
// calculate the maximum of two numbers
inline const int& max2( const int& a, const int& b ) noexcept
{
   return ( a >= b ) ? a : b;
}
// calculate the maximum of three numbers
inline const int& max3( const int& a, const int& b, const int& c ) noexcept
{
   return ( a >= b ) ? ( ( a >= c ) ? a : c ):
                       ( ( b >= c ) ? b : c );
}



class Stopwatch
{
public:
   void lap( std::string lap_name )
   {
      laps.insert_or_assign( lap_name, Clock::now() );
   }

   void reset() noexcept
   {
      laps.clear();
   }


   float dt( std::string lap1_name, std::string lap2_name )
   {
      auto p1_iter = laps.find( lap1_name );
      auto p2_iter = laps.find( lap2_name );

      auto p1 = p1_iter->second;
      auto p2 = p2_iter->second;
      return std::chrono::duration_cast<Resolution>( p1 - p2 ).count() / 1000.;
   }

private:
   using Clock = std::chrono::steady_clock;
   using Resolution = std::chrono::milliseconds;

   std::unordered_map< std::string, std::chrono::time_point<Clock> > laps;
};



// arguments for the Needleman-Wunsch algorithm variants
struct NwInput
{
   int* seqX;
   int* seqY;
   int* score;
   int* subst;

   int rows;
   int cols;
   // int substsz;

   // TODO: remove
   int insdelcost;
   // int inscost;
   // int delcost;
};

// results which the Needleman-Wunsch algorithm variants return
struct NwMetrics
{
   Stopwatch sw;
   float Tcpu;
   float Tgpu;
   std::vector<int> trace;
   unsigned hash;
};


using NwVariant = void (*)( NwInput& nw, NwMetrics& res );
void Nw_Cpu1_Row_St( NwInput& nw, NwMetrics& res );
void Nw_Cpu2_Diag_St( NwInput& nw, NwMetrics& res );
void Nw_Cpu3_DiagRow_St( NwInput& nw, NwMetrics& res );
void Nw_Cpu4_DiagRow_Mt( NwInput& nw, NwMetrics& res );
void Nw_Gpu3_DiagDiag_Coop( NwInput& nw, NwMetrics& res );


void Trace1_Diag( const NwInput& nw, NwMetrics& res );
inline void UpdateScore1_Simple(
   const int* const seqX,
   const int* const seqY,
   int* const score,
   const int* const subst,
   const int rows,
   const int cols,
   const int insdelcost,
   const int i,
   const int j )
   noexcept;
inline void UpdateScore2_Incremental(
   const int* const seqX,
   const int* const seqY,
   int* const score,
   const int* const subst,
   const int rows,
   const int cols,
   const int insdelcost,
   const int i,
   const int j )
   noexcept;





/*
./build -run  2048  2048 10
./build -run  6144  6144 10
./build -run 16384 16384 10
./build -run 22528 22528 10
*/

#include <cstdio>
#include <string>
#include <ctime>
#include <map>
#include "common.h"

// number of threads in warp
#define WARPSZ 32
// tile sizes for kernels A and B
// +   tile A should have one dimension be a multiple of the warp size for full memory coallescing
// +   tile B must have one dimension fixed to the number of threads in a warp
const int tileBx = 60;
const int tileBy = WARPSZ;


// call in case of invalid command line arguments
void Usage( char* argv[] )
{
   fprintf( stderr,
      "nw m n [cost]\n"
      "   m    - length of the first sequence\n"
      "   n    - length of the second sequence\n"
      "   cost - insert and delete cost (positive integer)\n" );
   fflush( stderr );
   exit( 0 );
}



// print one of the optimal matching paths to a file
void Trace( const NWArgs& nw, NWResult& res )
{
   // variable used to calculate the hash function
   // http://www.cse.yorku.ca/~oz/hash.html
   // the starting value is a magic constant
   unsigned hash = 5381;

   // for all elements on one of the optimal paths
   bool loop = true;
   for( int i = nw.rows-1, j = nw.cols-1;  loop;  )
   {
      // add the current element to the trace and hash
      int curr = el(nw.score,nw.adjcols, i,j);
      res.trace.push_back( curr );
      hash = ( ( hash<<5 ) + hash ) ^ curr;

      int max = std::numeric_limits<int>::min();   // maximum value of the up, left and diagonal neighbouring elements
      int dir = '-';                               // the current movement direction is unknown

      if( i > 0 && j > 0 && max < el(nw.score,nw.adjcols, i-1,j-1) ) { max = el(nw.score,nw.adjcols, i-1,j-1); dir = 'i'; }   // diagonal movement if possible
      if( i > 0          && max < el(nw.score,nw.adjcols, i-1,j  ) ) { max = el(nw.score,nw.adjcols, i-1,j  ); dir = 'u'; }   // up       movement if possible
      if(          j > 0 && max < el(nw.score,nw.adjcols, i  ,j-1) ) { max = el(nw.score,nw.adjcols, i  ,j-1); dir = 'l'; }   // left     movement if possible

      // move to the neighbour with the maximum value
      switch( dir )
      {
      case 'i': i--; j--; break;
      case 'u': i--;      break;
      case 'l':      j--; break;
      default:  loop = false; break;
      }
   }

   // reverse the trace, so it starts from the top-left corner of the matrix
   std::reverse( res.trace.begin(), res.trace.end() );
   // save the hash value
   res.hash = hash;
}



// main program
int main( int argc, char *argv[] )
{
   fflush( stdout );
   if( argc != 4 ) Usage( argv );

   // number of rows, number of columns and insdelcost
   int rows = atoi( argv[1] );
   int cols = atoi( argv[2] );
   int insdelcost = atoi( argv[3] );
   // add the padding (zeroth row and column) to the matrix
   rows++; cols++;
   // if the number of columns is less than the number of rows, swap them
   if( cols < rows ) { std::swap( rows, cols ); }

   // adjusted matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
   int adjrows = 1 + tileBy*ceil( float( rows-1 )/tileBy );
   int adjcols = 1 + tileBx*ceil( float( cols-1 )/tileBx );

   // allocate memory for the sequences which will be compared and the score matrix
   int* const seqX  = ( int* ) malloc( adjcols * sizeof( int ) );
   int* const seqY  = ( int* ) malloc( adjrows * sizeof( int ) );
   int* const score = ( int* ) malloc( adjrows*adjcols * sizeof( int ) );

   // if memory hasn't been allocated
   if( !seqX || !seqY || !score )
   {
      fprintf(stderr, "Error: memory allocation failed\n");
      fflush(stderr);

      // free allocated memory
      free( seqX ); free( seqY ); free( score );
      exit(1);
   }

   // seed the random generator
   unsigned int seed = time( NULL );
// unsigned int seed = 1605868371;
   srand( seed );

   // initialize the sequences A and B to random values in the range [1-10]
   // +   also initialize the padding with zeroes
   seqX[0] = 0;
   seqY[0] = 0;
   for( int j = 1; j < adjcols; j++ ) seqX[j] = ( j < cols ) ? 1 + rand() % 10 : 0;
   for( int i = 1; i < adjrows; i++ ) seqY[i] = ( i < rows ) ? 1 + rand() % 10 : 0;

   Stopwatch sw {};

   std::map<std::string, NWVariant> variants {
      { "Cpu1_Row", Cpu1_Row },
      { "Cpu2_Diag", Cpu2_Diag },
      { "Cpu3_DiagRow", Cpu3_DiagRow },
      { "Gpu3_DiagDiag", Gpu3_DiagDiag },
   };

   NWArgs nw {
      seqX,
      seqY,
      score,
      rows,
      cols,

      adjrows,
      adjcols,

      insdelcost,
   };

   // variables for storing the calculation hashes
   unsigned prevhash = 10;
   // if the test was successful
   bool firstIter = true;
   bool success = true;

   for( auto& variant_iter: variants )
   {
      const std::string& name = variant_iter.first;
      NWVariant& variant = variant_iter.second;

      NWResult res {};

      printf( "%-20s:   ", name.c_str() );
      fflush( stdout );

      variant( nw, res );

      Trace( nw, res );
      if( firstIter )
      {
         prevhash = res.hash;
      }
      else if( prevhash != res.hash )
      {
         success = false;
      }

      printf( "hash=%10u   Tcpu=%6.3fs   Tgpu=%6.3fs\n", res.hash, res.Tcpu, res.Tgpu );
      fflush( stdout );
   }

   // +   compare the implementations
   if( success ) printf( "TEST PASSED\n" );
   else          printf( "TEST FAILED\n" );
   fflush(stdout);

   // free allocated memory
   free( seqX ); free( seqY ); free( score );
}



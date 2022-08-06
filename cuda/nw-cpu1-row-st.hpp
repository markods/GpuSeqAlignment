#pragma once
#include "common.hpp"

// sequential cpu implementation of the Needleman Wunsch algorithm
void Nw_Cpu1_Row_St( NwInput& nw, NwMetrics& res )
{
   // start the timer
   res.sw.lap( "cpu-start" );


   // skip the first row and first column in the next calculation
   nw.rows--; nw.cols--;

   // initialize the first row and column of the score matrix
   for( int i = 0; i < 1+nw.rows; i++ ) el(nw.score,nw.cols, i,0) = -i*nw.insdelcost;
   for( int j = 0; j < 1+nw.cols; j++ ) el(nw.score,nw.cols, 0,j) = -j*nw.insdelcost;

   //  / / / / /
   //  / / / / /
   //  / / / / /
   for( int i = 0; i < nw.rows; i++ )
   for( int j = 0; j < nw.cols; j++ )
   {
      UpdateScore1_Simple( nw.seqX, nw.seqY, nw.score, nw.subst, nw.rows, nw.cols, nw.insdelcost, 1+i, 1+j );
   }

   // restore the original row and column count
   nw.rows++; nw.cols++;

   // stop the timer
   res.sw.lap( "cpu-end" );
   res.Tcpu = res.sw.dt( "cpu-end", "cpu-start" );
}








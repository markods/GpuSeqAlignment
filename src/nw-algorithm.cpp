#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <vector>
#include "nw-algorithm.hpp"


// align functions implemented in other files
NwStat NwAlign_Cpu1_Row_St( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Cpu2_Diag_St( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Cpu3_DiagRow_St( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Cpu4_DiagRow_Mt( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Gpu1_Diag_Ml( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Gpu2_DiagRow_Ml2K( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Gpu3_DiagDiag_Coop( NwParams& pr, NwInput& nw, NwResult& res );
NwStat NwAlign_Gpu4_DiagDiag_Coop2K( NwParams& pr, NwInput& nw, NwResult& res );
// void NwAlign_Gpu5_DiagDiagDiag_Ml( NwParams& pr, NwInput& nw, NwResult& res );

// traceback, hash and print functions implemented in other files
NwStat NwTrace1_Plain( const NwInput& nw, NwResult& res );
NwStat NwHash1_Plain( const int* const mat, const int rows, const int cols, unsigned& hash );
NwStat NwPrint1_Plain( std::ostream& os, const int* const mat, const int rows, const int cols );

// all algorithms
NwAlgorithmData algData
{
   /*algMap:*/ {
      { "Cpu1_Row_St",          { NwAlign_Cpu1_Row_St,          NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Cpu2_Diag_St",         { NwAlign_Cpu2_Diag_St,         NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Cpu3_DiagRow_St",      { NwAlign_Cpu3_DiagRow_St,      NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Cpu4_DiagRow_Mt",      { NwAlign_Cpu4_DiagRow_Mt,      NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Gpu1_Diag_Ml",         { NwAlign_Gpu1_Diag_Ml,         NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Gpu2_DiagRow_Ml2K",    { NwAlign_Gpu2_DiagRow_Ml2K,    NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Gpu3_DiagDiag_Coop",   { NwAlign_Gpu3_DiagDiag_Coop,   NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
      { "Gpu4_DiagDiag_Coop2K", { NwAlign_Gpu4_DiagDiag_Coop2K, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
   // { "Gpu5_DiagDiagDiag_Ml", { NwAlign_Gpu5_DiagDiagDiag_Ml, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain } },
   },
};


// conversion to object from json
void from_json( const json& j, NwSubstData& substData )
{
   j.at( "letterMap" ).get_to( substData.letterMap );
   j.at( "substMap"  ).get_to( substData.substMap  );
}
void from_json( const json& j, NwParamData& paramData )
{
   j.get_to( paramData.paramMap );
}
void from_json( const json& j, NwParams& params )
{
   j.get_to( params._params );
}
void from_json( const json& j, NwParam& param )
{
   j.get_to( param._values );
}
void from_json( const json& j, NwSeqData& seqData )
{
   j.at("substName").get_to( seqData.substName );
   j.at("indel"    ).get_to( seqData.indel     );
   j.at("repeat"   ).get_to( seqData.repeat    );
   j.at("seqList"  ).get_to( seqData.seqList   );
}


// conversion to json from object
void to_json( json& j, const NwSubstData& substData )
{
   j = json
   {
      { "letterMap", substData.letterMap },
      { "substMap",  substData.substMap  }
   };
}
void to_json( json& j, const NwParamData& paramData )
{
   j = json
   {
      { "paramMap", paramData.paramMap },
   };
}
void to_json( json& j, const NwParams& params )
{
   j = params._params;
}
void to_json( json& j, const NwParam& param )
{
   j = param._values;
}
void to_json( json& j, const NwSeqData& seqData )
{
   j = json
   {
      { "substName", seqData.substName },
      { "indel",     seqData.indel     },
      { "repeat",    seqData.repeat    },
      { "seqList",   seqData.seqList   }
   };
}



// conversion to csv from object
void resHeaderToCsv( std::ostream& os, const NwResData& resData )
{
   FormatFlagsGuard fg { os };
   os.fill(' ');

   os << "# " << std::setw(1) << std::left << "substFname: \""  << resData.substFname << "\"" << '\n';
   os << "# " << std::setw(1) << std::left << "paramFname: \""  << resData.paramFname << "\"" << '\n';
   os << "# " << std::setw(1) << std::left << "seqFname: \""    << resData.seqFname   << "\"" << '\n';
   os << "# ______________________________________________________________________________"   << '\n';

   os << std::setw(20) << std::left  << "algName" << ", ";
   os << std::setw( 2) << std::right << "iY" << ", ";
   os << std::setw( 2) << std::right << "iX" << ", ";
   os << std::setw( 2) << std::right << "iR" << ",   ";

   os << std::setw( 5) << std::right << "lenY" << ", ";
   os << std::setw( 5) << std::right << "lenX" << ",   ";

   os << std::setw(42) << std::left  << "algParams" << ",   ";

   os << std::setw( 1) << std::right << "errstep" << ", ";
   os << std::setw( 1) << std::right << "stat"    << ",   ";

   os << std::setw(10) << std::right << "score_hash" << ", ";
   os << std::setw(10) << std::right << "trace_hash" << ",   ";

   os << std::setw( 1) << std::left  << "       alloc,        cpy-dev,       init-hdr,         calc-1,       calc-2,       calc-3,       cpy-host,          total,     calc-sum" << '\n';
}
void to_csv( std::ostream& os, const NwResult& res )
{
   FormatFlagsGuard fg { os };
   {
      os.fill(' ');

      os << std::setw(20) << std::left  << res.algName << ", ";
      os << std::setw( 2) << std::right << res.iY << ", ";
      os << std::setw( 2) << std::right << res.iX << ", ";
      os << std::setw( 2) << std::right << res.iR << ",   ";

      os << std::setw( 5) << std::right << res.seqY_len << ", ";
      os << std::setw( 5) << std::right << res.seqX_len << ",   ";

      os << std::setw(42) << std::left; paramsToCsv( os, res.algParams ); os << ",   ";

      os << std::setw( 1) << std::right << res.errstep     << ", ";
      os << std::setw( 1) << std::right << int( res.stat ) << ",            ";

      os.fill('0');
      os << std::setw(10) << std::right << res.score_hash << ", ";
      os << std::setw(10) << std::right << res.trace_hash << ",   ";
   }
   fg.restore();
   to_csv( os, res.sw_align );
}
void paramsToCsv( std::ostream& os, const std::map<std::string, int>& paramMap )
{
   std::stringstream strs;
   {
      strs.fill(' ');
      
      strs << "\"";
      bool firstIter = true;
      for( auto iter = paramMap.begin();   iter != paramMap.end();   iter++ )
      {
         if( !firstIter ) { strs << " "; }
         else             { firstIter = false; }
         
         auto& paramName = iter->first;
         auto& paramValue = iter->second;
         strs << paramName << ":" << std::setw( 2) << std::right << paramValue;
      }
      strs << "\"";
   }

   os << strs.str();
}
void to_csv( std::ostream& os, const Stopwatch& sw )
{
   lapTimeToCsv( os, sw.get_or_default( "alloc"       ) ); os << ",   ";
   lapTimeToCsv( os, sw.get_or_default( "cpy-dev"  ) ); os << ",   ";
   lapTimeToCsv( os, sw.get_or_default( "init-hdr"    ) ); os << ",   ";
   lapTimeToCsv( os, sw.get_or_default( "calc-1"      ) ); os << ", ";
   lapTimeToCsv( os, sw.get_or_default( "calc-2"      ) ); os << ", ";
   lapTimeToCsv( os, sw.get_or_default( "calc-3"      ) ); os << ",   ";
   lapTimeToCsv( os, sw.get_or_default( "cpy-host" ) ); os << ",   ";

   float total = sw.total();
   float calc_total =
      sw.get_or_default( "calc-1" ) +
      sw.get_or_default( "calc-2" ) +
      sw.get_or_default( "calc-3" );

   lapTimeToCsv( os, total      ); os << ", ";
   lapTimeToCsv( os, calc_total );
}
void lapTimeToCsv( std::ostream& os, float lapTime )
{
   FormatFlagsGuard fg { os };

   os << std::fixed << std::setw(12) << std::setprecision(3) << std::setfill(' ') << lapTime;
}



// convert the sequence string to a vector using a character map
// + NOTE: add the header (zeroth) element if requested
std::vector<int> seqStrToVect( const std::string& str, const std::map<std::string, int>& map, const bool addHeader )
{
   // preallocate the requred amount of elements
   std::vector<int> vect { };

   // initialize the zeroth element if requested
   if( addHeader ) { vect.push_back( 0 ); }

   // for all characters of the string
   for( char c : str )
   {
      // add them to the vector
      std::string cs { c };
      int val = map.at( cs );
      vect.push_back( val );
   }

   return vect;
}


// get the current time as an ISO string
std::string IsoTime()
{
   auto now = std::chrono::system_clock::now();
   auto time = std::chrono::system_clock::to_time_t( now );

   std::stringstream strs;
   // https://en.cppreference.com/w/cpp/io/manip/put_time
   strs << std::put_time( std::localtime( &time ), "%Y%m%d_%H%M%S" );
   return strs.str();
}


// open output file stream
NwStat openOutFile( const std::string& path, std::ofstream& ofs )
{
   try
   {
      ofs.open( path, std::ios_base::out );
      if( !ofs )
      {
         return NwStat::errorIoStream;
      }
   }
   catch( const std::exception& ex )
   {
      return NwStat::errorIoStream;
   }
   
   return NwStat::success;
}







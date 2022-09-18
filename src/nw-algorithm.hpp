#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "json.hpp"
#include "common.hpp"
using json = nlohmann::ordered_json;

using NwAlignFn = NwStat (*)( NwParams& pr, NwInput& nw, NwResult& res );
using NwTraceFn = NwStat (*)( const NwInput& nw, NwResult& res );
using NwHashFn = NwStat (*)( const int* const mat, const int rows, const int cols, unsigned& hash );
using NwPrintFn = NwStat (*)( std::ostream& os, const int* const mat, const int rows, const int cols );


// the Needleman-Wunsch algorithm implementations
class NwAlgorithm
{
public:
   NwAlgorithm()
   {
      _alignFn = {};
      _traceFn = {};
      _hashFn = {};
      _printFn = {};

      _alignPr = {};
   }

   NwAlgorithm(
      NwAlignFn alignFn,
      NwTraceFn traceFn,
      NwHashFn hashFn,
      NwPrintFn printFn
   )
   {
      _alignFn = alignFn;
      _traceFn = traceFn;
      _hashFn = hashFn;
      _printFn = printFn;

      _alignPr = {};
   }

   void init( NwParams& alignPr )
   {
      _alignPr = alignPr;
   }

   NwParams& alignPr() { return _alignPr; }

   NwStat align( NwInput& nw, NwResult& res ) { return _alignFn( _alignPr, nw, res ); }
   NwStat trace( const NwInput& nw, NwResult& res ) { return _traceFn( nw, res ); }
   NwStat hash( const NwInput& nw, NwResult& res ) { return _hashFn( nw.score.data(), nw.adjrows, nw.adjcols, res.score_hash ); }
   NwStat print( std::ostream& os, const NwInput& nw ) { return _printFn( os, nw.score.data(), nw.adjrows, nw.adjcols ); }

private:
   NwAlignFn _alignFn;
   NwTraceFn _traceFn;
   NwHashFn _hashFn;
   NwPrintFn _printFn;

   NwParams _alignPr;
};

// algorithm map
struct NwAlgorithmData
{
   std::map< std::string, NwAlgorithm > algMap;
};
extern NwAlgorithmData algData;



// input file formats
struct NwSubstData
{
   std::map< std::string, int > letterMap;
   std::map< std::string, std::vector<int> > substMap;
};

struct NwParamData
{
   std::map< std::string, NwParams > paramMap;
};

struct NwSeqData
{
   std::string substName;
   int indel;
   // repeat each comparison this many times
   int repeat;
   // each sequence will be an int vector and have a header (zeroth) element
   std::vector<std::string> seqList;
};

// conversion to object from json
void from_json( const json& j, NwSubstData& substData );
void from_json( const json& j, NwParamData& paramData );
void from_json( const json& j, NwParams& params );
void from_json( const json& j, NwParam& param );
void from_json( const json& j, NwSeqData& seqData );

// conversion to json from object
void to_json( json& j, const NwSubstData& substData );
void to_json( json& j, const NwParamData& paramData );
void to_json( json& j, const NwParams& params );
void to_json( json& j, const NwParam& param );
void to_json( json& j, const NwSeqData& seqData );

// convert the sequence string to a vector using a character map
// + NOTE: add the header (zeroth) element if requested
std::vector<int> seqStrToVect( const std::string& str, const std::map<std::string, int>& map, const bool addHeader );



// get the current time as an ISO string
std::string IsoTime();

// open output file stream
NwStat openOutFile( const std::string& path, std::ofstream& ofs );

// read a json file into a variable
template< typename T >
NwStat readFromJson( const std::string& path, T& res )
{
   std::ifstream ifs;

   try
   {
      ifs.open( path, std::ios_base::in );
      if( !ifs )
      {
         return NwStat::errorIoStream;
      }
   }
   catch( const std::exception& ex )
   {
      return NwStat::errorIoStream;
   }

   try
   {
      // NOTE: the parser doesn't allow for trailing commas
      res = json::parse(
         ifs,
         /*callback*/ nullptr,
         /*allow_exceptions*/ true,
         /*ignore_comments*/ true
      );
   }
   catch( const std::exception& ex )
   {
      NwStat::errorInvalidFormat;
   }

   return NwStat::success;
}




#endif // INCLUDE_NW_ALGORITHM_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
// #include "llvm/TableGen/Error.h"
#include "TableGenBackends.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
using namespace llvm;

namespace {

class PreprocessorGuard {
  llvm::raw_ostream &OS;
  llvm::StringRef Guard;

public:
  explicit PreprocessorGuard(llvm::raw_ostream &OS, llvm::StringRef Guard)
      : OS(OS), Guard(Guard) {
    OS << "#ifdef " << Guard << '\n';
  }
  ~PreprocessorGuard() { OS << "#endif // " << Guard << '\n'; }

}; // class PreprocessorGuard

static std::string BuildTypeParameterList(ArrayRef<Record *> Parameters,
                                          const char *Format,
                                          StringRef Sep) {
  std::string Str;
  raw_string_ostream SOS(Str);
  llvm::interleave(
      Parameters, SOS,
      [&](Record *Param) {
        auto ParamName = Param->getValueAsString("name");
        SOS << llvm::formatv(Format, ParamName).str();
      },
      Sep);
  SOS.flush();
  return Str;
}

} // namespace

namespace clang {

void EmitPrecisionTypeList(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
  auto PrecisionTypes = Records.getAllDerivedDefinitions("PrecisionTypeDef");
  for (auto *TypeRecord : PrecisionTypes) {
    auto Name = TypeRecord->getValueAsString("name");
    auto Keyword = TypeRecord->getValueAsString("keyword");
    OS << "PRECISION_TYPE(" << Name << ", " << Name.lower() << ", " << Name.upper() << ", " << Keyword
       << ")\n";
    //  << "DEPENDENT_PRECISION_TYPE(Dependent" << Name << ")\n";
  }
}

void EmitPrecisionTypeNodes(llvm::RecordKeeper &Records,
                            llvm::raw_ostream &OS) {
  auto PrecisionTypes = Records.getAllDerivedDefinitions("PrecisionTypeDef");
  for (auto *TypeRecord : PrecisionTypes) {
    auto Name = TypeRecord->getValueAsString("name");
    OS << "def " << Name << "Type : TypeNode<Type>;\n";
    auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
    // PFIXME handle non parameterized types here
    OS << "def Dependent" << Name
       << "Type : TypeNode<Type>, AlwaysDependent;\n";
  }
}

void EmitPrecisionTypeProperties(llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS) {
  // Format arguments: name, properties, parameter names
  static const char *TypePropertiesTemplate = R"(
    let Class = {0}Type in {{
      {1}
      def : Creator<[{{
        return ctx.get{0}Type({3});
      }]>;
    }
    let Class = Dependent{0}Type in {{
      {2}
      def : Creator<[{{
        return ctx.getDependent{0}Type({3});
      }]>;
    }
  )";
  // Format arguments: name
  static const char *PropertyTemplate = R"(
    def : Property<"{0}", UInt32> {{
      let Read = [{{ node->get{0}() }];
    }
  )";
  static const char *DependentPropertyTemplate = R"(
    def : Property<"{0}", ExprRef> {{
      let Read = [{{ node->get{0}() }];
    }
  )";
  auto PrecisionTypes = Records.getAllDerivedDefinitions("PrecisionTypeDef");
  for (auto *TypeRecord : PrecisionTypes) {
    auto Name = TypeRecord->getValueAsString("name");
    // PFIXME handle non parameterized types here
    auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
    OS << llvm::formatv(
        TypePropertiesTemplate, Name,
        BuildTypeParameterList(Parameters, PropertyTemplate, "\n"),
        BuildTypeParameterList(Parameters, DependentPropertyTemplate, "\n"),
        BuildTypeParameterList(Parameters, "{0}", ", "));
  }
}

void EmitPrecisionParser(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
  auto PrecisionTypes = Records.getAllDerivedDefinitions("PrecisionTypeDef");
  //---------------------------------------------------------//
  // Parser
  //---------------------------------------------------------//

  // Format arguments: keyword, #parameters
  static const char *ParseDeclarationSpecifiersTemplate = R"(
    case tok::kw_{0}: {
      SmallVector<Expr *> Args;
      if (not ParsePrecisionTypeArguments({1}, Args))
        continue;
      isInvalid = DS.SetPrecisionType(Loc, TST_{2}, Args, PrevSpec, DiagID, Policy);
      ConsumedEnd = PrevTokLocation;
      break;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_PARSEDECLARATIONSPECIFIERS");
    for (auto *TypeRecord : PrecisionTypes) {
      auto name = TypeRecord->getValueAsString("name");
      auto keyword = TypeRecord->getValueAsString("keyword");
      auto parameters = TypeRecord->getValueAsListOfDefs("parameters");
      OS << llvm::formatv(ParseDeclarationSpecifiersTemplate, keyword,
                          parameters.size(), name.lower());
    }
  }
  // Format arguments: keyword, #parameters
  static const char *ParseCXXSimpleTypeSpecifiersTemplate = R"(
    case tok::kw_{0}: {
      SmallVector<Expr *> Args;
      if (not ParsePrecisionTypeArguments({1}, Args))
        DS.SetTypeSpecError();
      auto isInvalid = DS.SetPrecisionType(Loc, TST_{2}, Args, PrevSpec, DiagID, Policy);
      assert(not isInvalid);
      DS.SetRangeEnd(PrevTokLocation);
      DS.Finish(Actions, Policy);
      break;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_CXXSIMPLETYPESPECIFIERS");
    for (auto *TypeRecord : PrecisionTypes) {
      auto name = TypeRecord->getValueAsString("name");
      auto keyword = TypeRecord->getValueAsString("keyword");
      auto parameters = TypeRecord->getValueAsListOfDefs("parameters");
      OS << llvm::formatv(ParseCXXSimpleTypeSpecifiersTemplate, keyword,
                          parameters.size(), name.lower());
    }
  }
}

void EmitPrecisionAST(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
  auto PrecisionTypes = Records.getAllDerivedDefinitions("PrecisionTypeDef");
  const static char *ASTTypeDeclBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST Type Declarations
    ///
    //===----------------------------------------------------------------------===//
  )";
  // Format arguments: cppClassname, members, param list, getters
  static const char *ASTTypeTemplate = R"(
    class {0}Type final : public Type, public llvm::FoldingSetNode {{
      friend class ASTContext;
      {1}

    protected:
      {0}Type({2});

    public:
      {3}

      bool isSugared() const {{ return false; }
      QualType desugar() const {{ return QualType(this, 0); }

      void Profile(llvm::FoldingSetNodeID &ID) const {{
        Profile(ID, {4});
      }
      static void Profile(llvm::FoldingSetNodeID &ID, {2}) {{
        {5}
      }

      static bool classof(const Type *T) {{ return T->getTypeClass() == {0}; }
    };
  )";

  static const char *DependentASTTypeTemplate = R"(
    class Dependent{0}Type final : public Type, public llvm::FoldingSetNode {{
      friend class ASTContext;
      {1}

    protected:
      Dependent{0}Type({2});

    public:
      {3}

      bool isSugared() const {{ return false; }
      QualType desugar() const {{ return QualType(this, 0); }

      void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context) {{
        Profile(ID, Context, {4});
      }
      static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context, {2});

      static bool classof(const Type *T) {{ return T->getTypeClass() == Dependent{0}; }
    };
  )";

  OS << ASTTypeDeclBanner;
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTTYPE_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      assert(not Parameters.empty());
      auto Name = TypeRecord->getValueAsString("name");
      OS << llvm::formatv(
          ASTTypeTemplate, Name,
          BuildTypeParameterList(Parameters, "unsigned int {0};", "\n"),
          BuildTypeParameterList(Parameters, "unsigned int {0}", ", "),
          // BuildASTTypeParameterList(Parameters, "{0}({0}),", ", "),
          BuildTypeParameterList(
              Parameters, "unsigned int get{0}() const {{ return {0}; }", "\n"),
          BuildTypeParameterList(Parameters, "{0}", ", "),
          BuildTypeParameterList(Parameters, "ID.AddInteger({0});", "\n"));
      OS << llvm::formatv(
          DependentASTTypeTemplate, Name,
          BuildTypeParameterList(Parameters, "Expr *{0};", "\n"),
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "),
          BuildTypeParameterList(
              Parameters, "Expr *get{0}() const {{ return {0}; }", "\n"),
          BuildTypeParameterList(Parameters, "{0}", ", "));
    }
  }

  const static char *ASTTypeImplBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST Type Implementations
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTTypeImplBanner;
  static const char *ASTTypeImplTemplate = R"(
    {0}Type::{0}Type({1})
        : Type({0}, QualType{{}, TypeDependence::None), {2} {{}

    Dependent{0}Type::Dependent{0}Type({3})
        : Type(Dependent{0}, QualType{{}, toTypeDependence({4}->getDependence())), {2} {{}

    void Dependent{0}Type::Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                                   {3}) {{
      {5}
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTTYPE_IMPL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      assert(not Parameters.empty());
      auto Name = TypeRecord->getValueAsString("name");
      OS << llvm::formatv(
          ASTTypeImplTemplate, Name,
          BuildTypeParameterList(Parameters, "unsigned int {0}", ", "),
          BuildTypeParameterList(Parameters, "{0}({0})", ", "),
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "),
          Parameters.front()->getValueAsString("name"),
          BuildTypeParameterList(Parameters,
                                    "{0}->Profile(ID, Context, true);", "\n"));
    }
  }
  const static char *TypeLocDeclBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST TypeLoc Declarations
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << TypeLocDeclBanner;
  const static char *ASTTypeLocDeclTemplate = R"(
    class {0}TypeLoc final
        : public InheritingConcreteTypeLoc<TypeSpecTypeLoc, {0}TypeLoc,
                                          {0}Type> {{};
    class Dependent{0}TypeLoc final
        : public InheritingConcreteTypeLoc<TypeSpecTypeLoc, Dependent{0}TypeLoc,
                                          Dependent{0}Type> {{};
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTTYPELOC_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      OS << llvm::formatv(ASTTypeLocDeclTemplate, Name);
    }
  }
  const static char *ASTContextGetTypesDeclBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// ASTContext::Get*Type Decl
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTContextGetTypesDeclBanner;
  const static char *ASTContextGetTypeDeclTemplate = R"(
    QualType get{0}Type({1}) const;
    QualType getDependent{0}Type({2}) const;
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTCONTEXT_GETTYPE_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      OS << llvm::formatv(
          ASTContextGetTypeDeclTemplate, Name,
          BuildTypeParameterList(Parameters, "unsigned {0}", ", "),
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "));
    }
  }
  const static char *ASTContextGetTypeBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// ASTContext::Get*Type Impl
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTContextGetTypeBanner;
  const static char *ASTContextGetTypeImplTemplate = R"(
  QualType ASTContext::get{0}Type({1}) const {{
    llvm::FoldingSetNodeID ID;
    {0}Type::Profile(ID, {3});

    void *InsertPos = nullptr;
    if ({0}Type *Ty = {0}Types.FindNodeOrInsertPos(ID, InsertPos))
      return QualType(Ty, 0);

    auto *New = new (*this, alignof({0}Type)) {0}Type({3});
    {0}Types.InsertNode(New, InsertPos);
    Types.push_back(New);
    return QualType(New, 0);
  }

  QualType ASTContext::getDependent{0}Type({2}) const {{
    llvm::FoldingSetNodeID ID;
    Dependent{0}Type::Profile(ID, *this, {3});

    void *InsertPos = nullptr;
    if (Dependent{0}Type *Existing = Dependent{0}Types.FindNodeOrInsertPos(ID, InsertPos))
      return QualType(Existing, 0);

    auto *New = new (*this, alignof(Dependent{0}Type)) Dependent{0}Type({3});
    Dependent{0}Types.InsertNode(New, InsertPos);

    Types.push_back(New);
    return QualType(New, 0);
  }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTCONTEXT_GETTYPE_IMPL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      OS << llvm::formatv(
          ASTContextGetTypeImplTemplate, Name,
          BuildTypeParameterList(Parameters, "unsigned {0}", ", "),
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "),
          BuildTypeParameterList(Parameters, "{0}", ", "));
    }
  }
  const static char *ASTTypeVisitorBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST Type Visitor Declarations
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTTypeVisitorBanner;
  const static char *ASTTypeVisitorDeclTemplate = R"(
    DEF_TRAVERSE_TYPE({0}Type, {{})
    DEF_TRAVERSE_TYPE(Dependent{0}Type, {{
      {1}
    })
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTTYPEVISITOR_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          ASTTypeVisitorDeclTemplate, Name,
          BuildTypeParameterList(
              Parameters, "TRY_TO(TraverseStmt(T->get{0}()));", "\n"));
    }
  }
  const static char *ASTTypeLocVisitorBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST TypeLoc Visitor Declarations
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTTypeLocVisitorBanner;
  const static char *ASTTypeLocVisitorDeclTemplate = R"(
    DEF_TRAVERSE_TYPELOC({0}Type, {{})
    DEF_TRAVERSE_TYPELOC(Dependent{0}Type, {{
      {1}
    })
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTTYPELOCVISITOR_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          ASTTypeLocVisitorDeclTemplate, Name,
          BuildTypeParameterList(
              Parameters, "TRY_TO(TraverseStmt(TL.getTypePtr()->get{0}()));",
              "\n"));
    }
  }
  const static char *ASTEquivalentBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST Type Visitor Declarations
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTEquivalentBanner;
  const static char *ASTEquivalentTemplate = R"(
    case Type::{0}: {{
      const auto *Ty1 = cast<{0}Type>(T1);
      const auto *Ty2 = cast<{0}Type>(T2);

      if ({1})
        return false;
      break;
    }
    case Type::Dependent{0}: {{
      const auto *Ty1 = cast<Dependent{0}Type>(T1);
      const auto *Ty2 = cast<Dependent{0}Type>(T2);

      if ({2})
        return false;
      break;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTTEQUIVALENT_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          ASTEquivalentTemplate, Name,
          BuildTypeParameterList(Parameters,
                                    "Ty1->get{0}() != Ty2->get{0}()", " || "),
          BuildTypeParameterList(Parameters,
                                    "!IsStructurallyEquivalent(Context, "
                                    "Ty1->get{0}(), Ty2->get{0}())",
                                    " || "));
    }
  }
  const static char *ASTTypePrinterBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST TypePrinter
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTTypePrinterBanner;
  const static char *ASTTypePrinterTemplate = R"-(
    void TypePrinter::print{0}Before(const {0}Type *T, raw_ostream &OS) {{
      OS << "{0}(" << {1} << ")";
      spaceBeforePlaceHolder(OS);
    }
    void TypePrinter::print{0}After(const {0}Type *T, raw_ostream &OS) {{}

    void TypePrinter::printDependent{0}Before(const Dependent{0}Type *T, raw_ostream &OS) {{
      OS << "{0}(";
      {2};
      OS << ")";
      spaceBeforePlaceHolder(OS);
    }
    void TypePrinter::printDependent{0}After(const Dependent{0}Type *T, raw_ostream &OS) {{}

  )-";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_AST_TYPEPRINTER");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          ASTTypePrinterTemplate, Name,
          BuildTypeParameterList(Parameters, "T->get{0}()", " << \", \" << "),
          BuildTypeParameterList(
              Parameters, "T->get{0}()->printPretty(OS, nullptr, Policy);",
              "\nOS << \", \";\n"));
    }
  }
  const static char *MergeTypesImplBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// ASTContext MergeType
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << MergeTypesImplBanner;
  const static char *MergeTypesImplTemplate = R"(
    case Type::{0}: {{
      auto *LHSTy = LHS->castAs<{0}Type>();
      auto *RHSTy = RHS->castAs<{0}Type>();
      if ({1})
        return {{};
      return LHS;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTCONTEXT_MERGETYPE");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          MergeTypesImplTemplate, Name,
          BuildTypeParameterList(
              Parameters, "LHSTy->get{0}() != RHSTy->get{0}()", " || "));
    }
  }
  const static char *ASTMSCXXNameManglerBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST Type MicrosoftCXXNameMangler
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTMSCXXNameManglerBanner;
  const static char *ASTMSCXXNameManglerTemplate = R"(
    void MicrosoftCXXNameMangler::mangleType(const {0}Type *T, Qualifiers,
                                            SourceRange Range) {{
      llvm::SmallString<64> TemplateMangling;
      llvm::raw_svector_ostream Stream(TemplateMangling);
      MicrosoftCXXNameMangler Extra(Context, Stream);
      Stream << "?$";
      Extra.mangleSourceName("{0}");
      {1}
      mangleArtificialTagType(TagTypeKind::Struct, TemplateMangling, {{"__clang"});
    }

    void MicrosoftCXXNameMangler::mangleType(const Dependent{0}Type *T,
                                            Qualifiers, SourceRange Range) {{
      DiagnosticsEngine &Diags = Context.getDiags();
      unsigned DiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Error, "cannot mangle this Dependent{0} type yet");
      Diags.Report(Range.getBegin(), DiagID) << Range;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTMSCXXNAMEMANGLER_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          ASTMSCXXNameManglerTemplate, Name,
          BuildTypeParameterList(Parameters,
                                    "Stream << \"_\"; "
                                    "Extra.mangleIntegerLiteral(llvm::APSInt::"
                                    "getUnsigned(T->get{0}()));",
                                    "\n"));
      
    }
  }
  const static char *CXXNameManglerMangleTypeBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST CXXNameMangler::mangleType
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << CXXNameManglerMangleTypeBanner;
  const static char *CXXNameManglerMangleTypeTemplate = R"(
    void CXXNameMangler::mangleType(const {0}Type *T) {
      Out << "_{1}" << {2} << "_";
    }

    void CXXNameMangler::mangleType(const Dependent{0}Type *T) {
      Out << "_{1}";
      {3}
      Out << "_";
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_CXXNAMEMANGLER_MANGLETYPE");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto CppManglePrefix =
          TypeRecord->getValueAsOptionalString("cppManglePrefix");
      auto Prefix = CppManglePrefix.has_value()
                        ? CppManglePrefix.value()
                        : TypeRecord->getValueAsString("keyword").take_front();
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          CXXNameManglerMangleTypeTemplate, Name, Prefix,
          BuildTypeParameterList(Parameters, "T->get{0}()", " << '_' << "),
          BuildTypeParameterList(Parameters,
                                    "mangleExpression(T->get{0}());", "\n"));
    }
  }
  const static char *ASTNodeImporterBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// AST TypeLoc Visitor Declarations
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << ASTNodeImporterBanner;
  const static char *ASTNodeImporterTemplate = R"(
    ExpectedType clang::ASTNodeImporter::Visit{0}Type(const {0}Type *T) {{
      return Importer.getToContext().get{0}Type({1});
    }
    ExpectedType clang::ASTNodeImporter::VisitDependent{0}Type(const clang::Dependent{0}Type *T) {{
      {2}
      return Importer.getToContext().getDependent{0}Type({3});
    }
  )";
  const static char *ASTNodeImporterExprCheckTemplate = R"(
      ExpectedExpr {0}ExprOrErr = import(T->get{0}());
      if (!{0}ExprOrErr)
        return {0}ExprOrErr.takeError();
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_ASTNODEIMPORTER");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          ASTNodeImporterTemplate, Name,
          BuildTypeParameterList(Parameters, "T->get{0}()", ", "),
          BuildTypeParameterList(Parameters, ASTNodeImporterExprCheckTemplate,
                                 "\n"),
          BuildTypeParameterList(Parameters, "*{0}ExprOrErr", ", "));
    }
  }
  const static char *SemaMethodsDeclBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema methods decl
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaMethodsDeclBanner;
  const static char *SemaMethodsDeclTemplate = R"(
    QualType Build{0}Type({1}, SourceLocation Loc);
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_METHODS_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          SemaMethodsDeclTemplate, Name,
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "));
    }
  }
  const static char *SemaMethodsImplBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema methods implementation
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaMethodsImplBanner;
  const static char *SemaMethodsImplTemplate = R"(
    QualType Sema::Build{0}Type({1}, SourceLocation Loc) {{
      if ({2})
        return Context.getDependent{0}Type({3});
      {4}
      return Context.get{0}Type({5});
    }
  )";
  const static char *TypeArgCheckTemplate = R"(
    llvm::APSInt {0}Bits(32);
    if (VerifyIntegerConstantExpression({0}, &{0}Bits, AllowFold).isInvalid()) {{
      return QualType();
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_METHODS_IMPL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          SemaMethodsImplTemplate, Name,
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "),
          BuildTypeParameterList(Parameters,
                                    "{0}->isInstantiationDependent()", " || "),
          BuildTypeParameterList(Parameters, "{0}", ", "),
          BuildTypeParameterList(Parameters, TypeArgCheckTemplate, "\n"),
          BuildTypeParameterList(Parameters, "{0}Bits.getZExtValue()",
                                    ", "));
    }
  }
  const static char *SemaConvertDeclSpecToTypeBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema::ConvertDeclSpecToType
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaConvertDeclSpecToTypeBanner;
  const static char *SemaConvertDeclSpecToTypeTemplate = R"(
    case DeclSpec::TST_{0}: {
      auto ArgIter = DS.getPrecisionTypeArgs().begin();
      {2}
      Result = S.Build{1}Type({3}, DS.getBeginLoc());
      if (Result.isNull()) {
        S.Diag(DS.getTypeSpecTypeLoc(), diag::err_invalid_precision_type) << "{1}";
        declarator.setInvalidType(true);
      }
      break;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_CONVERTDECLSPECTOTYPE");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(SemaConvertDeclSpecToTypeTemplate, Name.lower(), Name,
                          BuildTypeParameterList(
                              Parameters, "Expr *{0} = *ArgIter++;", "\n"),
                          BuildTypeParameterList(Parameters, "{0}", ", "));
    }
  }
  const static char *SemaMarkUsedTemplateParametersBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema MarkUsedTemplateParameters
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaMarkUsedTemplateParametersBanner;
  const static char *SemaMarkUsedTemplateParametersTemplate = R"(
    case Type::Dependent{0}: {{
      auto *Ty = cast<Dependent{0}Type>(T);
      {1}
      break;
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_MARKUSEDTEMPLATEPARAMS");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          SemaMarkUsedTemplateParametersTemplate, Name,
          BuildTypeParameterList(
              Parameters, "MarkUsedTemplateParameters(Ctx, Ty->get{0}(), OnlyDeduced, Depth, Used);", "\n"));
    }
  }
  const static char *SemaTreeTransformTypesBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema TreeTransform::Transform*Type
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaTreeTransformTypesBanner;
  const static char *SemaTreeTransformTypesTemplate = R"(
    template <typename Derived>
    QualType TreeTransform<Derived>::Transform{0}Type(TypeLocBuilder &TLB,
                                                      {0}TypeLoc TL) {{
      const {0}Type *Ty = TL.getTypePtr();
      QualType Result = TL.getType();

      if (getDerived().AlwaysRebuild()) {{
        Result = getDerived().Rebuild{0}Type({1}, TL.getNameLoc());
        if (Result.isNull())
          return QualType();
      }

      {0}TypeLoc NewTL = TLB.push<{0}TypeLoc>(Result);
      NewTL.setNameLoc(TL.getNameLoc());
      return Result;
    }

    template <typename Derived>
    QualType TreeTransform<Derived>::TransformDependent{0}Type(TypeLocBuilder &TLB, Dependent{0}TypeLoc TL) {{
      const Dependent{0}Type *Ty = TL.getTypePtr();

      EnterExpressionEvaluationContext Unevaluated(
          SemaRef, Sema::ExpressionEvaluationContext::ConstantEvaluated);
      {2}

      QualType Result = TL.getType();

      if (getDerived().AlwaysRebuild() || {3}) {{
        Result = getDerived().RebuildDependent{0}Type({1}, TL.getNameLoc());
        if (Result.isNull())
          return QualType();
      }

      if (isa<Dependent{0}Type>(Result)) {{
        Dependent{0}TypeLoc NewTL = TLB.push<Dependent{0}TypeLoc>(Result);
        NewTL.setNameLoc(TL.getNameLoc());
      } else {{
        {0}TypeLoc NewTL = TLB.push<{0}TypeLoc>(Result);
        NewTL.setNameLoc(TL.getNameLoc());
      }
      return Result;
    }
  )";
  const static char *SemaTreeTransformTypeParamTemplate = R"(
    ExprResult {0} = getDerived().TransformExpr(Ty->get{0}());
    {0} = SemaRef.ActOnConstantExpression({0});
    if ({0}.isInvalid())
      return QualType();
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_TRANSFORM_TYPES");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          SemaTreeTransformTypesTemplate, Name,
          BuildTypeParameterList(Parameters, "Ty->get{0}()", ", "),
          BuildTypeParameterList(
              Parameters, SemaTreeTransformTypeParamTemplate, "\n"),
          BuildTypeParameterList(Parameters, "{0}.get() != Ty->get{0}()", " || "));
    }
  }
  const static char *SemaTreeTransformRebuildDeclBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema TreeTransform::Rebuld*Type Decl
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaTreeTransformRebuildDeclBanner;
  const static char *SemaTreeTransformRebuldDeclTemplate = R"(
    QualType Rebuild{0}Type({1}, SourceLocation Loc);
    QualType RebuildDependent{0}Type({2}, SourceLocation Loc);

  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_REBUILD_TYPES_DECL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          SemaTreeTransformRebuldDeclTemplate, Name,
          BuildTypeParameterList(Parameters, "unsigned {0}", ", "),
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "));
    }
  }
  const static char *SemaTreeTransformRebuildImplBanner = R"(
    //===----------------------------------------------------------------------===//
    ///
    /// Sema TreeTransform::Rebuld*Type Impl
    ///
    //===----------------------------------------------------------------------===//
  )";
  OS << SemaTreeTransformRebuildImplBanner;
  const static char *SemaTreeTransformRebuldImplTemplate = R"(
    template <typename Derived>
    QualType TreeTransform<Derived>::Rebuild{0}Type({1}, SourceLocation Loc) {{
      auto &Ctx = SemaRef.Context;
      auto UIntTy = Ctx.UnsignedIntTy;
      auto UIWidth = Ctx.getIntWidth(UIntTy);
      {3}
      return SemaRef.Build{0}Type({4}, Loc);
    }

    template <typename Derived>
    QualType TreeTransform<Derived>::RebuildDependent{0}Type({2}, SourceLocation Loc) {{
      return SemaRef.Build{0}Type({5}, Loc);
    }
  )";
  {
    PreprocessorGuard PPGuard(OS, "PRECISION_SEMA_REBUILD_TYPES_IMPL");
    for (auto *TypeRecord : PrecisionTypes) {
      auto Name = TypeRecord->getValueAsString("name");
      auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
      // PFIXME handle non parameterized types here
      OS << llvm::formatv(
          SemaTreeTransformRebuldImplTemplate, Name,
          BuildTypeParameterList(Parameters, "unsigned {0}", ", "),
          BuildTypeParameterList(Parameters, "Expr *{0}", ", "),
          BuildTypeParameterList(
              Parameters,
              "auto *{0}Lit = IntegerLiteral::Create(Ctx, llvm::APInt(UIWidth, "
              "{0}, true), UIntTy, Loc);",
              "\n"),
          BuildTypeParameterList(Parameters, "{0}Lit", ", "),
          BuildTypeParameterList(Parameters, "{0}", ", "));
    }
  }
}

void EmitPrecisionMLIRTypes(llvm::RecordKeeper &Records,
                            llvm::raw_ostream &OS) {
  auto PrecisionTypes = Records.getAllDerivedDefinitions("PrecisionTypeDef");
  const static char *ASTTypeDeclBanner = R"(
//===----------------------------------------------------------------------===//
// Precision Types
//===----------------------------------------------------------------------===//
  )";
  // Format arguments: cppClassname, summary, description, parameter list, assembly format
  static const char *PrecisionTypeTemplate = R"(
def Precision_{0}Type : Precision_Type<"{0}", "{1}"> {{
  let description = [{{
    {2}
  }];
  let parameters = (ins {3});
  let assemblyFormat = "`<` {4} `>`";
}
  )";
  OS << ASTTypeDeclBanner;
  for (auto *TypeRecord : PrecisionTypes) {
    auto Name = TypeRecord->getValueAsString("name");
    auto Mnemonic = TypeRecord->getValueAsString("mnemonic");
    auto Description = TypeRecord->getValueAsString("description");
    auto Parameters = TypeRecord->getValueAsListOfDefs("parameters");
    // PFIXME handle non parameterized types here
    assert(not Parameters.empty());
    OS << llvm::formatv(
        PrecisionTypeTemplate, Name, Mnemonic, Description,
        BuildTypeParameterList(Parameters, "\"int\":${0}", ", "),
        BuildTypeParameterList(Parameters, "${0}", " `,` "));
  }
}

} // namespace clang
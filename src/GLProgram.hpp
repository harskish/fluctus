#pragma once

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <GL/glew.h>
#include "math/float2.hpp"
#include "math/float3.hpp"
#include "math/matrix.hpp"

#define GL_SHADER_SOURCE(CODE) #CODE

using std::string;
using FireRays::float2;
using FireRays::float3;
using FireRays::float4;
using FireRays::matrix;

typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;
typedef void GLvoid;

class GLProgram
{
public:
	GLProgram		(const string& vertexSource, const string& fragmentSource);

	GLProgram		(const string& vertexSource,
					 GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const string& geometrySource,
					 const string& fragmentSource);

	~GLProgram(void);

	GLuint          getHandle(void) const { return m_glProgram; }
	GLint           getAttribLoc(const string& name) const;
	GLint           getUniformLoc(const string& name) const;

	void            use(void);

	static GLuint   createGLShader(GLenum type, const string& typeStr, const string& source);
	static void     linkGLProgram(GLuint prog);

	// Static collection of all compiled programs
	static GLProgram* get(const string &name);
	static void		  set(const string &name, GLProgram *prog);

	void             setUniform(int loc, int v) { if (loc >= 0) glUniform1i(loc, v); }
	void             setUniform(int loc, float v) { if (loc >= 0) glUniform1f(loc, v); }
	void             setUniform(int loc, double v) { if (loc >= 0) glUniform1d(loc, v); }
	void             setUniform(int loc, const float2& v) { if (loc >= 0) glUniform2f(loc, v.x, v.y); }
	void             setUniform(int loc, const float3& v) { if (loc >= 0) glUniform3f(loc, v.x, v.y, v.z); }
	void             setUniform(int loc, const matrix& v) { if (loc >= 0) glUniformMatrix4fv(loc, 1, false, &v.m00); }

	void			 setAttrib(int loc, int size, GLenum type, int stride, GLuint buffer, const void* pointer);
	void             setAttrib(int loc, int size, GLenum type, int stride, const void* pointer) { setAttrib(loc, size, type, stride, (GLuint)NULL, pointer); }
	void			 resetAttribs(void);

private:
	void            init(const string& vertexSource,
						 GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const string& geometrySource,
						 const string& fragmentSource);

private:
	GLProgram(const GLProgram&) = delete;
	GLProgram& operator=(const GLProgram&) = delete;

private:
	// Map that contains all compiled GLPrograms
	static std::map<string, GLProgram*> s_programs;
	
	int				m_numAttribs = 0;
	GLuint          m_glVertexShader;
	GLuint          m_glGeometryShader;
	GLuint          m_glFragmentShader;
	GLuint          m_glProgram;
};
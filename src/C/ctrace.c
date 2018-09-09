#include "include\Python\Python.h"
#include "include\numpy\arrayobject.h"

double det3(double *arr)
{
  double ret;

  ret = ((arr[0*3 + 0] * (arr[1*3 + 1]*arr[2*3 + 2] - arr[2*3 + 1]*arr[1*3 + 2]))
       - (arr[0*3 + 1] * (arr[1*3 + 0]*arr[2*3 + 2] - arr[2*3 + 0]*arr[1*3 + 2]))
       + (arr[0*3 + 2] * (arr[1*3 + 0]*arr[2*3 + 1] - arr[2*3 + 0]*arr[1*3 + 1])));

  return ret;
}

int comp_fcn(const void *elem1, const void *elem2)
{
  double e1 = (double) *((double*)elem1 + 2 );
  double e2 = (double) *((double*)elem2 + 2 );
  if (e1 > e2) return 1;
  if (e1 < e2) return -1;
  return 0;
}

void sort_by_3rd(double *arr, int rowNum, double *out_arr, int colNum)
{
//  double *out_arr;
//  out_arr = (double*) malloc(rowNum*3*sizeof(double));
  memcpy(out_arr, arr, rowNum*colNum*sizeof(double));

  qsort(out_arr, rowNum, colNum*sizeof(double), comp_fcn);

}


PyObject* ctrace_reflect(PyObject* self, PyObject* args)
{
  // In args
  PyObject *argPos_in = NULL, *argDir_in = NULL, *argMesh_in = NULL, *argNormal_in = NULL;
  int lastRefIdx_in;
  npy_intp *triShape;

  // args
  PyObject *Pos_in = NULL, *Dir_in = NULL, *Mesh_in = NULL, *Normal_in = NULL;
  double *Pos_data, *Dir_data, *Mesh_data, *Normal_data;

  PyObject *z_sol;  npy_intp z_sol_shape[1];   double *z_sol_data;

  PyObject *new_Dir = NULL, *new_Pos = NULL;
  double *new_Dir_data = NULL, *new_Pos_data = NULL;

  char isNewCollision = 0;  int triNum = 0; double *tri;
  double M[9];  double Mu[9]; double Mv[9]; double Mt[9];
  double detM = 0;
  double sol_uvt[3];
  double *sol_buffer = NULL, *sorted_sol = NULL;
  int solCnt = 0, closestIdx = 0;
  double alpha_norm = 0;

  //ret
  PyObject *retval = NULL;

  z_sol_shape[0] = 3;
  if (!PyArg_ParseTuple(args, "OOOOi", &argPos_in, &argDir_in, &argMesh_in, &argNormal_in, &lastRefIdx_in))
    return NULL;

  Pos_in = PyArray_FROM_OTF(argPos_in, NPY_DOUBLE, NPY_IN_ARRAY);
  if (Pos_in == NULL) return NULL;
  Dir_in = PyArray_FROM_OTF(argDir_in, NPY_DOUBLE, NPY_IN_ARRAY);
  if (Dir_in == NULL) goto fail;
  Mesh_in = PyArray_FROM_OTF(argMesh_in, NPY_DOUBLE, NPY_IN_ARRAY);
  if (Mesh_in == NULL) goto fail;
  Normal_in = PyArray_FROM_OTF(argNormal_in, NPY_DOUBLE, NPY_IN_ARRAY);
  if (Normal_in == NULL) goto fail;
  new_Dir = PyArray_SimpleNew(1, PyArray_DIMS(Dir_in), NPY_DOUBLE);
  new_Pos = PyArray_SimpleNew(1, PyArray_DIMS(Pos_in), NPY_DOUBLE);

  Pos_data = (double*) PyArray_DATA(Pos_in);
  Dir_data = (double*) PyArray_DATA(Dir_in);
  Mesh_data = (double*) PyArray_DATA(Mesh_in);
  Normal_data = (double*) PyArray_DATA(Normal_in);
  new_Pos_data = (double*) PyArray_DATA(new_Pos);
  new_Dir_data = (double*) PyArray_DATA(new_Dir);

  triShape = PyArray_DIMS(Mesh_in);

  z_sol = PyArray_SimpleNew(1, z_sol_shape, NPY_DOUBLE);
  z_sol_data = (double*) PyArray_DATA(z_sol);
  if (!((Dir_data[2] < 1.0E-05) && (Dir_data[2] > -1.0E-05)))
  {
    z_sol_data[0] = Pos_data[0] - Dir_data[0]*(Pos_data[2]/Dir_data[2]);
    z_sol_data[1] = Pos_data[1] - Dir_data[1]*(Pos_data[2]/Dir_data[2]);
    z_sol_data[2] = (-1)*(Pos_data[2]/Dir_data[2]);
  }
  else
  {
    z_sol_data[0] = -1.0;
    z_sol_data[1] = -1.0;
    z_sol_data[2] = -1.0;
  }

  for(triNum = 0; triNum < triShape[0]; triNum++)
  {
    tri = (Mesh_data + 9*triNum);
    ////
    M[0*3 + 0] = tri[1*3 + 0] - tri[2*3 + 0]; M[0*3 + 1] = tri[0*3 + 0] - tri[2*3 + 0]; M[0*3 + 2] = (-1)*Dir_data[0];
    M[1*3 + 0] = tri[1*3 + 1] - tri[2*3 + 1]; M[1*3 + 1] = tri[0*3 + 1] - tri[2*3 + 1]; M[1*3 + 2] = (-1)*Dir_data[1];
    M[2*3 + 0] = tri[1*3 + 2] - tri[2*3 + 2]; M[2*3 + 1] = tri[0*3 + 2] - tri[2*3 + 2]; M[2*3 + 2] = (-1)*Dir_data[2];
    ////
    detM = det3(M);
    if (!((detM < 1.0E-05) && (detM > -1.0E-05)))
    {
      ////replace column 0 in M by constant b
      Mu[0*3 + 0] = Pos_data[0] - tri[2*3 + 0]; Mu[0*3 + 1] = tri[0*3 + 0] - tri[2*3 + 0]; Mu[0*3 + 2] = (-1)*Dir_data[0];
      Mu[1*3 + 0] = Pos_data[1] - tri[2*3 + 1]; Mu[1*3 + 1] = tri[0*3 + 1] - tri[2*3 + 1]; Mu[1*3 + 2] = (-1)*Dir_data[1];
      Mu[2*3 + 0] = Pos_data[2] - tri[2*3 + 2]; Mu[2*3 + 1] = tri[0*3 + 2] - tri[2*3 + 2]; Mu[2*3 + 2] = (-1)*Dir_data[2];
      ////
      ////replace column 1 in M by constant b
      Mv[0*3 + 0] = tri[1*3 + 0] - tri[2*3 + 0]; Mv[0*3 + 1] = Pos_data[0] - tri[2*3 + 0]; Mv[0*3 + 2] = (-1)*Dir_data[0];
      Mv[1*3 + 0] = tri[1*3 + 1] - tri[2*3 + 1]; Mv[1*3 + 1] = Pos_data[1] - tri[2*3 + 1]; Mv[1*3 + 2] = (-1)*Dir_data[1];
      Mv[2*3 + 0] = tri[1*3 + 2] - tri[2*3 + 2]; Mv[2*3 + 1] = Pos_data[2] - tri[2*3 + 2]; Mv[2*3 + 2] = (-1)*Dir_data[2];
      ////
      ////replace column 2 in M by constant b
      Mt[0*3 + 0] = tri[1*3 + 0] - tri[2*3 + 0]; Mt[0*3 + 1] = tri[0*3 + 0] - tri[2*3 + 0]; Mt[0*3 + 2] = Pos_data[0] - tri[2*3 + 0];
      Mt[1*3 + 0] = tri[1*3 + 1] - tri[2*3 + 1]; Mt[1*3 + 1] = tri[0*3 + 1] - tri[2*3 + 1]; Mt[1*3 + 2] = Pos_data[1] - tri[2*3 + 1];
      Mt[2*3 + 0] = tri[1*3 + 2] - tri[2*3 + 2]; Mt[2*3 + 1] = tri[0*3 + 2] - tri[2*3 + 2]; Mt[2*3 + 2] = Pos_data[2] - tri[2*3 + 2];
      ////
      sol_uvt[0] = det3(Mu)/detM;  //solve for u
      sol_uvt[1] = det3(Mv)/detM;  //solve for v
      sol_uvt[2] = det3(Mt)/detM;  //solve for t

      if ((sol_uvt[2] > 0) && (triNum != lastRefIdx_in) && (0 <= sol_uvt[0])
          && (sol_uvt[0] <= 1) && (0 <= sol_uvt[1]) && (sol_uvt[1] <= 1)
          && ((sol_uvt[0] + sol_uvt[1]) <= 1))
      {
        isNewCollision = 1;
        solCnt++;
        sol_buffer = (double*) realloc(sol_buffer, solCnt*4*sizeof(double));
        sol_buffer[4*(solCnt-1) + 0] = sol_uvt[0]; //u
        sol_buffer[4*(solCnt-1) + 1] = sol_uvt[1]; //v
        sol_buffer[4*(solCnt-1) + 2] = sol_uvt[2]; //t
        sol_buffer[4*(solCnt-1) + 3] = (double) (triNum); //for later lut use
      }
    }
  }

  sorted_sol = (double*) malloc(solCnt * 4 * sizeof(double));
  sort_by_3rd(sol_buffer, solCnt, sorted_sol, 4);
  closestIdx = (int) (sorted_sol[4*0 + 3] + 0.4);
  tri = (Mesh_data + 9*closestIdx);

  if (isNewCollision && ((z_sol_data[2] > sorted_sol[2]) || (z_sol_data[2] <= 0)))
  {
    // found a coliding triangle and triangle appears before z=0 plane
    new_Pos_data[0] = tri[2*3 + 0] + (tri[1*3 + 0] - tri[2*3 + 0])*sorted_sol[0] + (tri[0*3 + 0] - tri[2*3 + 0])*sorted_sol[1];
    new_Pos_data[1] = tri[2*3 + 1] + (tri[1*3 + 1] - tri[2*3 + 1])*sorted_sol[0] + (tri[0*3 + 1] - tri[2*3 + 1])*sorted_sol[1];
    new_Pos_data[2] = tri[2*3 + 2] + (tri[1*3 + 2] - tri[2*3 + 2])*sorted_sol[0] + (tri[0*3 + 2] - tri[2*3 + 2])*sorted_sol[1];

    alpha_norm = 2.0*((Normal_data[closestIdx*3 + 0]*Dir_data[0] + Normal_data[closestIdx*3 + 1]*Dir_data[1] + Normal_data[closestIdx*3 + 2]*Dir_data[2])
                    /(Normal_data[closestIdx*3 + 0]*Normal_data[closestIdx*3 + 0] + Normal_data[closestIdx*3 + 1]*Normal_data[closestIdx*3 + 1] + Normal_data[closestIdx*3 + 2]*Normal_data[closestIdx*3 + 2]));

    new_Dir_data[0] = Dir_data[0] - (alpha_norm*Normal_data[closestIdx*3 + 0]);
    new_Dir_data[1] = Dir_data[1] - (alpha_norm*Normal_data[closestIdx*3 + 1]);
    new_Dir_data[2] = Dir_data[2] - (alpha_norm*Normal_data[closestIdx*3 + 2]);

    retval = Py_BuildValue("OOi", new_Pos, new_Dir, closestIdx);
    Py_DECREF(new_Pos);
    Py_DECREF(new_Dir);
    Py_DECREF(z_sol);
    Py_DECREF(Pos_in);
    Py_DECREF(Dir_in);
    Py_DECREF(Mesh_in);
    Py_DECREF(Normal_in);
    free(sorted_sol);
    free(sol_buffer);

    return retval;
  }
  else if (z_sol_data[2] > 0)
  {
    new_Pos_data[0] = z_sol_data[0];
    new_Pos_data[1] = z_sol_data[1];
    new_Pos_data[2] = 0.0;

    new_Dir_data[0] = Dir_data[0];
    new_Dir_data[1] = Dir_data[1];
    new_Dir_data[2] = (-1)*Dir_data[2];

    retval = Py_BuildValue("OOi", new_Pos, new_Dir, *(PyArray_DIMS(Mesh_in)));
    Py_DECREF(new_Pos);
    Py_DECREF(new_Dir);
    Py_DECREF(z_sol);
    Py_DECREF(Pos_in);
    Py_DECREF(Dir_in);
    Py_DECREF(Mesh_in);
    Py_DECREF(Normal_in);
    free(sorted_sol);
    free(sol_buffer);

    return retval;
  }
  else
  {
    Py_DECREF(new_Pos);
    Py_DECREF(new_Dir);
    Py_DECREF(z_sol);
    Py_DECREF(Pos_in);
    Py_DECREF(Dir_in);
    Py_DECREF(Mesh_in);
    Py_DECREF(Normal_in);
    free(sorted_sol);
    free(sol_buffer);
    Py_INCREF(Py_None);

    return Py_None;
  }



  fail:
  Py_XDECREF(Pos_in);
  Py_XDECREF(Dir_in);
  Py_XDECREF(Mesh_in);
  Py_XDECREF(Normal_in);
  return NULL;


}


PyObject* ctrace_sort(PyObject *self, PyObject *args)
{
  PyObject *arg = NULL;
  PyObject *arr = NULL;
  PyObject *out_arr = NULL;
  double *arr_data, *out_data;
  npy_intp *shape;
  npy_intp nd;

  if (!PyArg_ParseTuple(args, "O", &arg))
    return NULL;
  arr = PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_IN_ARRAY);
  shape = PyArray_DIMS(arr);
  nd = PyArray_NDIM(arr);
  arr_data = (double*) PyArray_DATA(arr);


  out_arr = PyArray_SimpleNew(nd, shape, NPY_DOUBLE);
  if (out_arr == NULL)
  {
    Py_DECREF(arr);
    return NULL;
  }

  out_data = (double*) PyArray_DATA(out_arr);

  sort_by_3rd(arr_data, shape[0], out_data, shape[1]);

  Py_DECREF(arr);
  return out_arr;

}

PyObject* ctrace_add(PyObject* self, PyObject* args)
{
  PyObject *arg1 = NULL, *arg2 = NULL;
  PyObject *arr1 = NULL, *arr2 = NULL;
  PyObject *out;
  npy_intp nd1 = 0, nd2 = 0, *shape1, *shape2;
  npy_intp len = 1;
  double *data_in_1, *data_in_2, *data_out;
  int i = 0;


  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
    return NULL;

  arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (arr1 == NULL) return NULL;
  arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (arr2 == NULL) goto fail;

  nd1 = PyArray_NDIM(arr1); nd2 = PyArray_NDIM(arr2);
  shape1 = PyArray_DIMS(arr1); shape2 = PyArray_DIMS(arr2);

  out = PyArray_SimpleNew(nd1, shape1, NPY_DOUBLE);

  for(i = 0; i < nd1; i++)  len *= shape1[i];

  data_in_1 = (double*) PyArray_DATA(arr1);
  data_in_2 = (double*) PyArray_DATA(arr2);
  data_out = (double*) PyArray_DATA(out);

  for(i = 0; i < len; i++)
  {
    *(data_out+i) = *(data_in_1 + i) + *(data_in_2 + i);
  }

  Py_DECREF(arr1);
  Py_DECREF(arr2);

  return out;

  fail:
  Py_XDECREF(arr1);
  Py_XDECREF(arr2);
  return NULL;
}

static PyMethodDef globalMethodsTable[] =
{
  /*{"PythonFuncName", cfunction,
    ARGS_MODIFIERS,
    "Doc String"},*/
    {"addArr", ctrace_add,
     METH_VARARGS,
     "Adds elements of two arrays arbytrary by dims of first arguments"},
    {"sort", ctrace_sort,
     METH_VARARGS,
     "sorts array by last column"},
    {"reflect", ctrace_reflect,
     METH_VARARGS,
     "reflect(argPos_in, argDir_in, argMesh_in, argNormal_in, lastRefIdx_in) -> retTuple\n retTuple = (newPos, newDir, lastrefidx)"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initctrace(void)
{
  (void) Py_InitModule("ctrace", globalMethodsTable);
  import_array();
}
